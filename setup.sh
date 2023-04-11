# Optiions: MINDsmall_train, MINDsmall_dev, MINDsmall_train, MINDlarge_train, MINDlarge_dev, MINDlarge_test
DATASET=MINDlarge_train
mkdir data/ 
mkdir data/index/ data/$DATASET
curl -o data/$DATASET.zip -L https://mind201910small.blob.core.windows.net/release/$DATASET.zip 
unzip data/$DATASET.zip -d data/$DATASET
rm data/$DATASET.zip
pip install -r requirements.txt
