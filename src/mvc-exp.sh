cd /home/xuezhe/guanzc/mvc/src/models
python -m train.py -c mnist_contrast --n_epochs 200 --model_config__optimizer_config__learning_rate 0.0001
echo "mvc experiment has done!" | mail -s "mvc-exp" 884691896@qq.com
