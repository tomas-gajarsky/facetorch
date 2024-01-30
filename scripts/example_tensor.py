import hydra
import torch
from facetorch import FaceAnalyzer
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../conf", config_name="tensor.config")
def main(cfg: DictConfig) -> None:

    tensor = torch.load(cfg.path_tensor, map_location=torch.device(cfg.analyzer.device))
    # tensor = torch.randint(0, 255, (3, 512, 512), dtype=torch.uint8).to(cfg.analyzer.device)

    analyzer = FaceAnalyzer(cfg.analyzer)

    response = analyzer.run(
        tensor=tensor,
        batch_size=cfg.batch_size,
        fix_img_size=cfg.fix_img_size,
        return_img_data=cfg.return_img_data,
        include_tensors=cfg.include_tensors,
        path_output=cfg.path_output,
    )
    print(response)


if __name__ == "__main__":
    main()
