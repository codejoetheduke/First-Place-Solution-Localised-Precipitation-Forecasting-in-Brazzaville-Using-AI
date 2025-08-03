TRAIN_PATH = "data/Train_data.csv"
TEST_PATH = "data/Test_data.csv"

DIFF_COLUMNS = ['RH2M', 'QV2M', 'T2MDEW', 'PS']
DROP_COLUMNS_CAT = ['Set','DATE', 'ID', 'DY']
DROP_COLUMNS_LGB = ['Set','DATE', 'ID']

N_SPLITS = 10