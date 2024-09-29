from titanic.data_steps import *

TRAIN_DATA = "s3://titanic-542/data/raw/train.csv"
TEST_DATA = "s3://titanic-542/data/raw/test.csv"

OUT_TRAIN = "s3://titanic-542/data/preprocessed/train.csv"
OUT_TEST = "s3://titanic-542/data/preprocessed/test.csv"

df = load_data(TRAIN_DATA, TEST_DATA)
df = clean_cabin_deck(df)
df = replace_line(df)
df = ppr_tickets(df)
df = do_replaces(df)
df = clean_names(df)
df = clean_titles(df)
df = add_kid_col(df)
df = add_old_col(df)
df = add_alone_col(df)
df = ppr_sex(df)
df = ppr_fare(df)
df = ppr_embarked(df)
df = onehot(df)
split_and_save(df, OUT_TRAIN, OUT_TEST)