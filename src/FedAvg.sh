# python federated_main.py --model=cnn --dataset=mnist --gpu=0 --iid=0 --num_users 50 --local_ep 10 --rm_type 0 --epochs 5 --rm_step 5 --verbose 0 --iter_num 10
# python federated_main.py --model=cnn --dataset=mnist --gpu=0 --iid=0 --num_users 50 --local_ep 10 --rm_type 1 --epochs 5 --rm_step 5 --verbose 0 --iter_num 10
# python federated_main.py --model=cnn --dataset=mnist --gpu=0 --iid=0 --num_users 50 --local_ep 10 --rm_type 2 --epochs 5 --rm_step 5 --verbose 0 --iter_num 10

python federated_main.py --model=cnn --dataset=mnist --gpu=0 --iid=0 --num_users 50 --local_ep 10 --rm_type 2 --epochs 5 --rm_step 5 --verbose 0 --iter_num 10 --plot_only

