import hydra
from facetorch import FaceAnalyzer
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    analyzer = FaceAnalyzer(cfg.analyzer)

    response = analyzer.run(
        image_source=cfg.path_image,
        batch_size=cfg.batch_size,
        fix_img_size=cfg.fix_img_size,
        return_img_data=cfg.return_img_data,
        include_tensors=cfg.include_tensors,
        path_output=cfg.path_output,
    )
    print(response)


if __name__ == "__main__":
    main()
