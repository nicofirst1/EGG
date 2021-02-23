from egg.zoo.coco_game.analysis.interaction_analysis.accuracy import accuracy_analysis
from egg.zoo.coco_game.analysis.interaction_analysis.language import language_analysis
from egg.zoo.coco_game.analysis.interaction_analysis.utils import path_parser

if __name__ == "__main__":
    interaction_path, out_dir = path_parser()

    accuracy_analysis(interaction_path, out_dir)
    language_analysis(interaction_path, out_dir)
