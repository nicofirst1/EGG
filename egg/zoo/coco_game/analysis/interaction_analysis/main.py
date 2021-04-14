from egg.zoo.coco_game.analysis.interaction_analysis.Analysis import Analysis
from egg.zoo.coco_game.analysis.interaction_analysis.utils import path_parser


def perform_all(analysis: Analysis):
    analysis.update_analysis()
    analysis.update_infos()
    analysis.plot_cm()
    analysis.plot_correlations()
    analysis.plot_2d_language_tensor()
    analysis.add_infos()


if __name__ == "__main__":
    interaction_path, out_dir = path_parser()

    class_out_dir = out_dir.joinpath("Classes")
    superclass_out_dir = out_dir.joinpath("SuperClasses")

    analysis_path = out_dir.joinpath(".helpers")

    class_out_dir.mkdir(exist_ok=True)
    superclass_out_dir.mkdir(exist_ok=True)
    analysis_path.mkdir(exist_ok=True)

    class_analysis = Analysis(interaction_path, class_out_dir, analysis_path, "class")
    perform_all(class_analysis)

    superclass_analysis = Analysis(interaction_path, superclass_out_dir, analysis_path, "superclass")
    perform_all(superclass_analysis)

    a = 1
