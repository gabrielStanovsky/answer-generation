### Run in root directory of repository ###

#########################################
### Sets up a virtualenv, activates the env, and installs pip packages
##########################################
virtualenv -p python3 answer-generation-env
source answer-generation-env/bin/activate

# Install the CUDA10.0 version of torch first (since allennlp installs the CUDA8.0 version of torch)
pip install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl
pip install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl

# Install the all pip packages
pip install -r requirements.txt

##########################################
### Downloads the data into the raw directory
##########################################
mkdir -p raw_data
cd raw_data/

# Download CosmosQA
mkdir cosmosqa
cd cosmosqa
wget https://github.com/wilburOne/cosmosqa/raw/master/data/train.csv
wget https://github.com/wilburOne/cosmosqa/raw/master/data/valid.csv
cd ..

# Download DROP
wget https://s3-us-west-2.amazonaws.com/allennlp/datasets/drop/drop_dataset.zip
unzip drop_dataset.zip
mv drop_dataset drop
rm drop_dataset.zip

# Download MCScript
wget https://my.hidrive.com/api/sharelink/download?id=DhAhE8B5
unzip 'download?id=DhAhE8B5'
rm 'download?id=DhAhE8B5'
rm -r __MACOSX
mv MCScript mcscript

# Download NarrativeQA
git clone https://github.com/deepmind/narrativeqa.git
cd narrativeqa
./download_stories.sh
cd ../

# Download QUOREF
wget https://quoref-dataset.s3-us-west-2.amazonaws.com/train_and_dev/quoref-train-dev-v0.1.zip
unzip quoref-train-dev-v0.1.zip
mv quoref-train-dev-v0.1 quoref
rm quoref-train-dev-v0.1.zip

# Download RACE by filling out the online form

# Download SocialIQA
mkdir socialiqa
cd socialiqa
wget https://maartensap.github.io/social-iqa/data/socialIQa_v1.4.tgz
tar -zxvf socialIQa_v1.4.tgz
cd ..