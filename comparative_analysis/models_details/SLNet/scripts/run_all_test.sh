EPOCH=1
FPS_METHOD='pointops2'  # [pytorch3d, pointops2, pytorch]
KNN_METHOD='pytorch3d'  # [pytorch3d, pytorch]

# modelnet
python tasks/cls_modelnet.py --epoch "$EPOCH" --fps_method "$FPS_METHOD" --knn_method "$KNN_METHOD"

# scanobject
python tasks/cls_scanobject.py --epoch "$EPOCH" --fps_method "$FPS_METHOD" --knn_method "$KNN_METHOD"

# shapenet
python tasks/partseg_shapenet.py --epoch "$EPOCH" --fps_method "$FPS_METHOD" --knn_method "$KNN_METHOD" # --workers 6 --batch_size 16 --test_batch_size 8

# fewshot
python tasks/cls_fewshot.py --epoch "$EPOCH" --fps_method "$FPS_METHOD" --knn_method "$KNN_METHOD"

# eval model
python tasks/eval_model.py

# attention map
python tasks/attention_map.py