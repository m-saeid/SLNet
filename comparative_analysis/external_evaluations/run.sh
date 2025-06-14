# APES
cd /home/iris/Desktop/Saeid_2080__/other_codes/APES
# local
bash utils/single_gpu_train.sh configs/apes/apes_cls_local-modelnet-200epochs.py
bash utils/single_gpu_train.sh configs/apes/apes_cls_local-modelnet-200epochs.py
bash utils/single_gpu_train.sh configs/apes/apes_cls_local-modelnet-200epochs.py
# global
bash utils/single_gpu_train.sh configs/apes/apes_cls_local-modelnet-200epochs.py
bash utils/single_gpu_train.sh configs/apes/apes_cls_local-modelnet-200epochs.py
bash utils/single_gpu_train.sh configs/apes/apes_cls_local-modelnet-200epochs.py


# PointNet
#cd /home/iris/Desktop/Saeid_2080/other_codes/Pointnet_Pointnet2/Pointnet_Pointnet2_pytorch-master
#python train_classification.py --model pointnet_cls --log_dir pointnet_cls_1
#python train_classification.py --model pointnet_cls --log_dir pointnet_cls_2
#python train_classification.py --model pointnet_cls --log_dir pointnet_cls_3


# PointNet++
#cd /home/iris/Desktop/Saeid_2080__/other_codes/Pointnet_Pointnet2/Pointnet_Pointnet2_pytorch-master
# ssg
#python train_classification.py --model pointnet2_cls_ssg --log_dir pointnet2_cls_ssg_1
#python train_classification.py --model pointnet2_cls_ssg --log_dir pointnet2_cls_ssg_2
#python train_classification.py --model pointnet2_cls_ssg --log_dir pointnet2_cls_ssg_3
# msg
#python train_classification.py --model pointnet2_cls_msg --log_dir pointnet2_cls_msg_1
#python train_classification.py --model pointnet2_cls_msg --log_dir pointnet2_cls_msg_2
#python train_classification.py --model pointnet2_cls_msg --log_dir pointnet2_cls_msg_3


# DGCNN
#cd /home/iris/Desktop/Saeid_2080/other_codes/dgcnn/dgcnn-master/pytorch
#python main.py --exp_name=dgcnn_1024_1 --model=dgcnn --num_points=1024 --k=20 --use_sgd=True
#python main.py --exp_name=dgcnn_1024_2 --model=dgcnn --num_points=1024 --k=20 --use_sgd=True
#python main.py --exp_name=dgcnn_1024_3 --model=dgcnn --num_points=1024 --k=20 --use_sgd=True

# CurveNet
#cd /home/iris/Desktop/Saeid_2080/other_codes/CurveNet/CurveNet-main/core
#python3 main_cls.py --exp_name=curvenet_cls_1
#python3 main_cls.py --exp_name=curvenet_cls_2
#python3 main_cls.py --exp_name=curvenet_cls_3

# PointMLP-elite
#cd /home/iris/Desktop/Saeid_2080/other_codes/pointMLP/pointMLP-pytorch-main/classification_ModelNet40
#python main.py --model pointMLPElite --msg pointMLPElite_1
#python main.py --model pointMLPElite --msg pointMLPElite_2
#python main.py --model pointMLPElite --msg pointMLPElite_3
# PointMLP
#python main.py --model pointMLP --msg pointMLP_1 --batch_size 28
#python main.py --model pointMLP --msg pointMLP_1 --batch_size 28
#python main.py --model pointMLP --msg pointMLP_1 --batch_size 28