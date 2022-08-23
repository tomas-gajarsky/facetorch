import os
from typing import List, Tuple, Union
import torch
import numpy as np
from codetiming import Timer
from facetorch.base import BaseDownloader, BaseUtilizer
from facetorch.datastruct import ImageData
from facetorch.logger import LoggerJsonFile
from torchvision import transforms

logger = LoggerJsonFile().logger


class Lmk3DMeshPose(BaseUtilizer):
    def __init__(
        self,
        transform: transforms.Compose,
        device: torch.device,
        optimize_transform: bool,
        downloader_meta: BaseDownloader,
        image_size: int = 120,
    ):
        """Initializes the Lmk3DMeshPose class. This class is used to convert the face parameter vector to 3D landmarks, mesh and pose.

        Args:
            transform (Compose): Composed Torch transform object.
            device (torch.device): Torch device cpu or cuda object.
            optimize_transform (bool): Whether to optimize the transform.
            downloader_meta (BaseDownloader): Downloader for metadata.
            image_size (int): Standard size of the face image.

        """
        super().__init__(transform, device, optimize_transform)

        self.downloader_meta = downloader_meta
        self.image_size = image_size
        if not os.path.exists(self.downloader_meta.path_local):
            self.downloader_meta.run()

        self.meta = torch.load(self.downloader_meta.path_local)

        for key in self.meta.keys():
            if isinstance(self.meta[key], torch.Tensor):
                self.meta[key] = self.meta[key].to(self.device)

        self.keypoints = self.meta["keypoints"]
        # PCA basis for shape, expression, texture
        self.w_shp = self.meta["w_shp"]
        self.w_exp = self.meta["w_exp"]
        # param_mean and param_std are used for re-whitening
        self.param_mean = self.meta["param_mean"]
        self.param_std = self.meta["param_std"]
        # mean values
        self.u_shp = self.meta["u_shp"]
        self.u_exp = self.meta["u_exp"]

        self.u = self.u_shp + self.u_exp
        self.w = torch.cat((self.w_shp, self.w_exp), dim=1)
        # base vector for landmarks
        self.w_base = self.w[self.keypoints]
        self.w_norm = torch.linalg.norm(self.w, dim=0)
        self.w_base_norm = torch.linalg.norm(self.w_base, dim=0)
        self.u_base = self.u[self.keypoints].reshape(-1, 1)
        self.w_shp_base = self.w_shp[self.keypoints]
        self.w_exp_base = self.w_exp[self.keypoints]
        self.dim = self.w_shp.shape[0] // 3

    @Timer("Lmk3DMeshPose.run", "{name}: {milliseconds:.2f} ms", logger.debug)
    def run(self, data: ImageData) -> ImageData:
        """Runs the Lmk3DMeshPose class functionality - convert the face parameter vector to 3D landmarks, mesh and pose.

        Adds the following attributes to the data object:

        - landmark [[y, x, z], 68 (points)]
        - mesh [[y, x, z], 53215 (points)]
        - pose (Euler angles [yaw, pitch, roll] and translation [y, x, z])

        Args:
            data (ImageData): ImageData object containing most of the data including the predictions.

        Returns:
            ImageData: ImageData object containing lmk3d, mesh and pose.
        """
        for count, face in enumerate(data.faces):
            assert "align" in face.preds.keys(), "align key not found in face.preds"
            param = face.preds["align"].logits

            roi_box = [face.loc.x1, face.loc.y1, face.loc.x2, face.loc.y2]

            landmarks = self._compute_sparse_vert(param, roi_box, transform_space=True)
            vertices = self._compute_dense_vert(param, roi_box, transform_space=True)
            angles, translation = self._compute_pose(param, roi_box)

            data.faces[count].preds["align"].other["lmk3d"] = landmarks
            data.faces[count].preds["align"].other["mesh"] = vertices
            data.faces[count].preds["align"].other["pose"] = dict(
                angles=angles, translation=translation
            )

        return data

    def _matrix2angle_corr(self, re: torch.Tensor) -> List[float]:
        """Converts a rotation matrix to angles.

        Args:
            re (torch.Tensor): Rotation matrix.

        Returns:
            List[float]: List of angles.
        """
        pi = torch.tensor(np.pi).to(self.device)
        if re[2, 0] != 1 and re[2, 0] != -1:
            x = torch.asin(re[2, 0])
            y = torch.atan2(
                re[1, 2] / torch.cos(x),
                re[2, 2] / torch.cos(x),
            )
            z = torch.atan2(
                re[0, 1] / torch.cos(x),
                re[0, 0] / torch.cos(x),
            )

        else:  # Gimbal lock
            z = 0
            if re[2, 0] == -1:
                x = pi / 2
                y = z + torch.atan2(re[0, 1], re[0, 2])
            else:
                x = -pi / 2
                y = -z + torch.atan2(-re[0, 1], -re[0, 2])

        rx, ry, rz = (
            float((x * 180 / pi).item()),
            float((y * 180 / pi).item()),
            float((z * 180 / pi).item()),
        )

        return [rx, ry, rz]

    def _parse_param(self, param: torch.Tensor):
        """Parses the parameter vector.

        Args:
            param (torch.Tensor): Parameter vector.

        Returns:
            Tuple[torch.Tensor]
        """
        p_ = param[:12].reshape(3, 4)
        pe = p_[:, :3]
        offset = p_[:, -1].reshape(3, 1)
        alpha_shp = param[12:52].reshape(40, 1)
        alpha_exp = param[52:62].reshape(10, 1)
        return pe, offset, alpha_shp, alpha_exp

    def _param2vert(
        self, param: torch.Tensor, dense: bool = False, transform_space: bool = True
    ) -> torch.Tensor:
        """Parses the parameter vector into a dense or sparse vertex representation.

        Args:
            param (torch.Tensor): Parameter vector.
            dense (bool): Whether to return a dense or sparse vertex representation.
            transform_space (bool): Whether to transform the vertex representation to the original space.

        Returns:
            torch.Tensor: Dense or sparse vertex representation.
        """

        def _reshape_fortran(_x, shape):
            if len(_x.shape) > 0:
                _x = _x.permute(*reversed(range(len(_x.shape))))
            return _x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))

        if param.shape[0] == 62:
            param_ = param * self.param_std[:62] + self.param_mean[:62]
        else:
            raise RuntimeError("length of params mismatch")

        pe, offset, alpha_shp, alpha_exp = self._parse_param(param_)

        if dense:
            he = (
                self.u + self.w_shp @ alpha_shp.float() + self.w_exp @ alpha_exp.float()
            )
            he = _reshape_fortran(he, (3, -1))
            vertex = pe.float() @ he + offset
            if transform_space:
                # transform to image coordinate space
                vertex[1, :] = self.image_size + 1 - vertex[1, :]

        else:
            he = (
                self.u_base
                + self.w_shp_base @ alpha_shp.float()
                + self.w_exp_base @ alpha_exp.float()
            )
            he = _reshape_fortran(he, (3, -1))
            vertex = pe.float() @ he + offset
            if transform_space:
                # transform to image coordinate space
                vertex[1, :] = self.image_size + 1 - vertex[1, :]

        return vertex

    def _p2srt(
        self, param: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Applies matrix norm to the parameter vector.

        Args:
            param (torch.Tensor): Parameter vector.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        t3d = param[:, 3]
        r1 = param[0:1, :3]
        r2 = param[1:2, :3]
        se = (torch.linalg.norm(r1) + torch.linalg.norm(r2)) / 2.0
        r1 = r1 / torch.linalg.norm(r1)
        r2 = r2 / torch.linalg.norm(r2)
        r3 = torch.cross(r1, r2)
        re = torch.cat((r1, r2, r3), 0)
        return se, re, t3d

    def _parse_pose(
        self, param: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """Parses the parameter vector to pose data.

        Args:
            param (torch.Tensor): Parameter vector.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]: Pose data.
        """
        param = param * self.param_std[:62] + self.param_mean[:62]
        param = param[:12].reshape(3, -1)  # camera matrix
        _, rem, t3d = self._p2srt(param)
        pe = torch.cat((rem, t3d.reshape(3, -1)), 1)  # without scale
        pose = self._matrix2angle_corr(rem)  # yaw, pitch, roll
        return pe, pose, t3d

    def _compute_vertices(
        self,
        param: torch.Tensor,
        roi_bbox: Tuple[int, int, int, int],
        dense: bool,
        transform_space: bool = True,
    ) -> torch.Tensor:
        """Predict the vertices of the face given the parameter vector.

        Args:
            param (torch.Tensor): Parameter vector.
            roi_bbox (Tuple[int, int, int, int]): Bounding box of the face.
            dense (bool): Whether to return a dense or sparse vertex representation.
            transform_space (bool): Whether to transform the vertex representation to the original space.

        Returns:
            torch.Tensor: Dense or sparse vertex representation.
        """
        vertex = self._param2vert(param, dense=dense, transform_space=transform_space)
        sx, sy, ex, ey = roi_bbox
        scale_x = (ex - sx) / self.image_size
        scale_y = (ey - sy) / self.image_size
        vertex[0, :] = vertex[0, :] * scale_x + sx
        vertex[1, :] = vertex[1, :] * scale_y + sy

        s = (scale_x + scale_y) / 2
        vertex[2, :] *= s

        return vertex

    def _compute_sparse_vert(
        self,
        param: torch.Tensor,
        roi_box: Tuple[int, int, int, int],
        transform_space: bool = False,
    ) -> torch.Tensor:
        """Predict the sparse vertex representation of the face given the parameter vector.

        Args:
            param (torch.Tensor): Parameter vector.
            roi_box (Tuple[int, int, int, int]): Bounding box of the face.
            transform_space (bool): Whether to transform the vertex representation to the original space.

        Returns:
            torch.Tensor: Sparse vertex representation.

        """
        vertex = self._compute_vertices(
            param, roi_box, dense=False, transform_space=transform_space
        )
        return vertex

    def _compute_dense_vert(
        self,
        param: torch.Tensor,
        roi_box: Tuple[int, int, int, int],
        transform_space: bool = False,
    ) -> torch.Tensor:
        """Predict the dense vertex representation of the face given the parameter vector.

        Args:
            param (torch.Tensor): Parameter vector.
            roi_box (Tuple[int, int, int, int, int]): Bounding box of the face.
            transform_space (bool): Whether to transform the vertex representation to the original space.

        Returns:
            torch.Tensor: Dense vertex representation.
        """
        vertex = self._compute_vertices(
            param, roi_box, dense=True, transform_space=transform_space
        )
        return vertex

    def _compute_pose(
        self,
        param: torch.Tensor,
        roi_bbox: Tuple[int, int, int, int],
        ret_mat: bool = False,
    ) -> Union[torch.Tensor, Tuple[List[float], torch.Tensor]]:
        """Predict the pose of the face given the parameter vector.

        Args:
            param (torch.Tensor): Parameter vector.
            roi_bbox (Tuple[int, int, int, int, int]): Bounding box of the face.
            ret_mat (bool): Whether to return the rotation matrix.

        Returns:
            Union[torch.Tensor]: Pose of the face.
        """
        pe, angles, t3d = self._parse_pose(param)

        sx, sy, ex, ey = roi_bbox
        scale_x = (ex - sx) / self.image_size
        scale_y = (ey - sy) / self.image_size
        t3d[0] = t3d[0] * scale_x + sx
        t3d[1] = t3d[1] * scale_y + sy

        if ret_mat:
            return pe
        return angles, t3d
