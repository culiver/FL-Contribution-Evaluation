# # For training
# python federated_main.py --model=cnn --dataset=mnist --gpu=0 --iid=0 --num_users 50 --local_ep 10 --rm_type 0 --epochs 5 --rm_step 5 --verbose 0 --iter_num 10 --trainer ka
# python federated_main.py --model=cnn --dataset=mnist --gpu=0 --iid=0 --num_users 50 --local_ep 10 --rm_type 1 --epochs 5 --rm_step 5 --verbose 0 --iter_num 10 --trainer ka
# python federated_main.py --model=cnn --dataset=mnist --gpu=0 --iid=0 --num_users 50 --local_ep 10 --rm_type 2 --epochs 5 --rm_step 5 --verbose 0 --iter_num 10 --trainer ka

# # For plot
# python federated_main.py --model=cnn --dataset=mnist --gpu=0 --iid=0 --num_users 50 --local_ep 10 --rm_type 2 --epochs 5 --rm_step 5 --verbose 0 --iter_num 10 --plot_only --trainer ka

# # For training
# python federated_main.py --model=cnn --dataset=mnist --gpu=0 --iid=0 --num_users 10 --local_ep 2 --rm_type 0 --epochs 30 --rm_step 1 --verbose 0 --iter_num 10 --trainer ka --ka_ep=20
# python federated_main.py --model=cnn --dataset=mnist --gpu=0 --iid=0 --num_users 10 --local_ep 2 --rm_type 1 --epochs 30 --rm_step 1 --verbose 0 --iter_num 10 --trainer ka --ka_ep=20
# python federated_main.py --model=cnn --dataset=mnist --gpu=0 --iid=0 --num_users 10 --local_ep 2 --rm_type 2 --epochs 30 --rm_step 1 --verbose 0 --iter_num 10 --trainer ka --ka_ep=20

# # # For debugging
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --iid=0 --num_users 3 --local_ep 2 --rm_type 1 --epochs 10 --rm_step 1 --verbose 0 --iter_num 1 --trainer ka  --ka_ep=1

# # For training single iteration
# python federated_main.py --model=cnn --dataset=mnist --gpu=0 --iid=0 --num_users 10 --local_ep 2 --rm_type 0 --epochs 30 --rm_step 1 --verbose 0 --iter_num 1 --trainer ka --ka_ep=20
# python federated_main.py --model=cnn --dataset=mnist --gpu=0 --iid=0 --num_users 10 --local_ep 2 --rm_type 1 --epochs 30 --rm_step 1 --verbose 0 --iter_num 1 --trainer ka --ka_ep=20
# python federated_main.py --model=cnn --dataset=mnist --gpu=0 --iid=0 --num_users 10 --local_ep 2 --rm_type 2 --epochs 30 --rm_step 1 --verbose 0 --iter_num 1 --trainer ka --ka_ep=20
