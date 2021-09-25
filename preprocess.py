import utils


# utils.copy_folder_as_256x256("D:/ILSVRC2010_images_train")
# utils.copy_folder_as_256x256("D:/val")
# utils.map_folders_to_numbers_json("D:/ILSVRC2010_images_train")
utils.put_val_into_synset_folders("D:/val_256", "D:/devkit-1.0/data/ILSVRC2010_validation_ground_truth.txt", "D:/devkit-1.0/data/synsets.csv")