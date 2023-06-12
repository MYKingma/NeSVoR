cd ./S1 
git clone --depth 1 https://github.com/daviddmc/NeSVoR.git
mkdir ./NeSVoR/nesvor/checkpoints
wget -O ./NeSVoR/nesvor/checkpoints/SVoRT_v2.pt https://zenodo.org/record/7486938/files/checkpoint_v2.pt?download=1
docker build --force-rm -t nesvor-s1 .
docker run --name nesvor_container --gpus all nesvor-s1:latest /bin/bash install.sh
docker commit nesvor_container nesvor-s1:latest
rm -rf ./NeSVoR/
docker rm nesvor_container

cd ../S2
docker build --force-rm -t nesvor-s2 .
