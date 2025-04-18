# DISTILLATION_LR_STUDENT


your_inference_directory/
├── run_inference.py
├── student.py
├── student.pth
├── teacher_weights.pth
├── model/
│   └── ops.py
└── archs/
    └── rrdbnet_arch.py

FOR INFERENCE 
python run_inference.py \
    --lr_image input_image.png \
    --student_pth student.pth \
    --teacher_pth teacher_weights.pth \
    --scale 4 \
    --output_dir ./inference_output \
    --device cuda \
    --teacher_num_feat 64 \
    --teacher_num_block 23 \
    # Add --hr_image /path/to/hr.png if you have ground truth
    # Add --student_multi_scale or --student_group if needed for your student.pth



    COPY PASTE ABOVE IN TERMINEL IN  FOLDER IN THE CREATED ANACONDA 
