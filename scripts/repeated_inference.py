import hydra
from facetorch import FaceAnalyzer
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    analyzer = FaceAnalyzer(cfg.analyzer)

    for i in range(5):
        analyzer.run(
            path_image=cfg.path_image,
            batch_size=cfg.batch_size,
            fix_img_size=cfg.fix_img_size,
        )
        analyzer.run(
            path_image=cfg.path_image_2,
            batch_size=cfg.batch_size,
            fix_img_size=cfg.fix_img_size,
        )


if __name__ == "__main__":
    main()
