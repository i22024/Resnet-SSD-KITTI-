from utils import create_data_lists

if __name__ == '__main__':
    create_data_lists(voc07_path='./data/VOC2007', # specify your data root
                      voc12_path='./data/VOC2012',
                      output_folder='./')
