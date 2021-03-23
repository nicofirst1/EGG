import pandas as pd


def check_no_intersection(train, test):
    train_data = set(train['image_id'])
    test_data = set(test['image_id'])

    same_data = train_data.intersection(test_data)

    assert len(same_data) == 0

    train_data = set(train['id_true'])
    test_data = set(test['id_true'])

    same_data = train_data.intersection(test_data)
    assert len(same_data) == 0

    train_data = set(train['id_false'])
    test_data = set(test['id_false'])

    same_data = train_data.intersection(test_data)
    assert len(same_data) == 0


def check_sequence(seq):

    seq=list(seq)
    guess = 1
    max_len = len(seq) // 2
    for x in range(2, max_len):
        if seq[0:x] == seq[x:2 * x]:
            return x

    assert guess==1

def check_same_cats(train, test):
    train_data = list(train['cat_true']) + list(train['cat_false'])
    test_data = list(test['cat_true']) + list(test['cat_false'])

    train_data = set(train_data)
    test_data = set(test_data)

    assert train_data == test_data

if __name__ == '__main__':

    train_file = "../../data_debug_train.csv"
    test_file = "../../data_debug_test.csv"

    train = pd.read_csv(train_file, sep=r'\s*,\s*')
    test = pd.read_csv(test_file, sep=r'\s*,\s*')

    check_no_intersection(train, test)
    check_same_cats(train, test)

    check_sequence(train['indice'])
    check_sequence(test['indice'])

    check_sequence(train['image_id'])
    check_sequence(test['image_id'])

    print("Everything ok")
    a=12