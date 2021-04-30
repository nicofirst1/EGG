import pandas as pd

def check_interactions(interactions):

    interactions=[pd.read_csv(x) for x in interactions]

    col1 = "True Class"
    col2 = "True SuperClass"

    for idx in range(len(interactions)):
        pdi=interactions[idx]

        for jdx in range(idx+1, len(interactions)):
            pdj=interactions[jdx]

            assert all(pdi[col1] == pdj[col1])
            assert all(pdi[col2] == pdj[col2])

    print("Everything is fine")






if __name__ == "__main__":
    seg_path = "/home/dizzi/Desktop/EGG/egg/zoo/coco_game/Logs/seg/runs/interactions.csv"
    both_path = "/home/dizzi/Desktop/EGG/egg/zoo/coco_game/Logs/both/runs/interactions.csv"

    interaction_paths = [seg_path, both_path]

    out_path = "/home/dizzi/Desktop/EGG/egg/zoo/coco_game/Logs/"

    ca = check_interactions(interaction_paths)


    a = 1
