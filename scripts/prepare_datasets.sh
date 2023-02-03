# RefCOCO 
# gdown --id 1_Qkfq7aUVUfNocX70BwH5p-1a_Vzr1Vw
# unzip refcoco.zip 

# Refer-YouTube-VOS
gdown --id 1xjAwiPZColmGCKUYtMXO-Tc5Zzm1a-sJ
unzip meta_expressions_test.zip -d datasets/ref_ytvos_2021
gdown --id 1yH9YywIBzNfepwLLqzxFHXQxa89tjrkq
unzip -q valid.zip -d datasets/ref_ytvos_2021/
gdown --id 13Eqw0gVK-AO5B-cqvJ203mZ2vzWck9s4
unzip -q train.zip -d datasets/ref_ytvos_2021
rm -rf train.zip
rm -rf valid.zip
cd datasets/ref_ytvos_2021
gdown --id 1FJqOZ_tFqpXbxy2azphrhKDnakg1I-0x 
gdown --id 1eAhCCc-j1QpSk9gefdfcS2_8y38g6275 
gdown --id 1ve6H-HYUYwvlae5wiNlCJYX0Z0x2jBQf
cd ../..


# wget http://images.cocodataset.org/zips/train2014.zip 
# unzip -q train2014.zip 
# wget https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip
# wget https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip
# wget https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip
# unzip -q refcoco.zip
# unzip -q refcoco+.zip
# unzip -q refcocog.zip

# cd ../../

# python utils/convert_refexp_to_coco.py