from egg.zoo.coco_game.tests.test_bbox import test_bbox
from egg.zoo.coco_game.tests.test_data import test_data_mods
from egg.zoo.coco_game.tests.test_losses import test_loss_mods
from egg.zoo.coco_game.tests.test_receiver_mods import test_receiver_mods
from egg.zoo.coco_game.tests.test_sender_mods import test_sender_mods

if __name__ == "__main__":
    test_bbox()
    test_data_mods()
    test_loss_mods()
    test_receiver_mods()
    test_sender_mods()
