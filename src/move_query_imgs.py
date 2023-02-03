from utils import move_imgs


if __name__ == '__main__':
    source = '../datasets/holidays_dataset/database'
    destination = '../datasets/holidays_dataset/query_imgs'
    move_imgs(source, destination)