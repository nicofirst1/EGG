import pathlib
from pathlib import PosixPath

from egg.zoo.coco_game.analysis.interaction_analysis.classes.Analysis import Analysis
from egg.zoo.coco_game.analysis.interaction_analysis.classes.ComparedAnalysis import ComparedAnalysis
from egg.zoo.coco_game.analysis.interaction_analysis.classes.JoinedAnalysis import JoinedAnalysis
from egg.zoo.coco_game.analysis.interaction_analysis.utils import define_out_dir


def perform_all(analysis: Analysis):
    analysis.plot_cm()
    analysis.plot_correlations()
    analysis.plot_2d_language_tensor()
    analysis.add_infos()


def add_readme(out_dir_path: PosixPath):
    readme_path = out_dir_path.joinpath("README.md")

    with open(readme_path, "w+") as f:
        f.write("In the following dir you can find two folders.\n"
                "Both have the same types of analysis but performed on different datasets\n"
                "- *Classes* : uses the information regarding an object class, i.e. dog, spoon, motorbike\n"
                "- *SuperClasses* : uses the information regarding an object superclass, i.e. animal, appliance, vehicle\n"
                )


def generate_analysis(interaction_path, generate=True):
    interaction_path = pathlib.Path(interaction_path)
    out_dir = define_out_dir(interaction_path)

    add_readme(out_dir)

    class_out_dir = out_dir.joinpath("Classes")
    superclass_out_dir = out_dir.joinpath("SuperClasses")
    joined_out_dir = out_dir.joinpath("Joined")
    analysis_path = out_dir.joinpath(".helpers")

    class_out_dir.mkdir(exist_ok=True)
    superclass_out_dir.mkdir(exist_ok=True)
    joined_out_dir.mkdir(exist_ok=True)
    analysis_path.mkdir(exist_ok=True)

    class_analysis = Analysis(interaction_path, class_out_dir, analysis_path, "_class")

    superclass_analysis = Analysis(interaction_path, superclass_out_dir, analysis_path, "_superclass")

    if generate:
        perform_all(class_analysis)
        perform_all(superclass_analysis)

    joined_anal = JoinedAnalysis(interaction_path, joined_out_dir, analysis_path, class_analysis, superclass_analysis)
    joined_anal.same_superclass_sequence()
    joined_anal.add_meningful_data()
    joined_anal.add_readme()

    return joined_anal


def compare_anal(interaction_paths, out_path, generate=True):
    joined = []
    for ip in interaction_paths:
        j = generate_analysis(ip, generate=generate)
        joined.append(j)

    ca = ComparedAnalysis(joined, out_path)
    return ca


if __name__ == "__main__":
    seg_path = "/home/dizzi/Desktop/EGG/egg/zoo/coco_game/Logs/seg/runs/interactions.csv"
    both_path = "/home/dizzi/Desktop/EGG/egg/zoo/coco_game/Logs/both/runs/interactions.csv"

    interaction_paths=[seg_path, both_path]

    out_path = "/home/dizzi/Desktop/EGG/egg/zoo/coco_game/Logs/"

    ca=compare_anal(interaction_paths, out_path, generate=False)

    a = 1
