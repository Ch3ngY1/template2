1. conda create -n xxxxxxxxxx python=3.9
2. conda install -c conda-forge mamba
3. mamba install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
4. pip install -r requirements.txt
5. pip install -e .

should change:
main.py:
* --save_folder
* --save_log

config.cfg.py
* self.args.save_log
* self.args.save_folder






需要跑的实验：
pnet:
1. Pretrain
2. Seg
    * 7训练(vote=0)，验证使用7个（1-6+vote）
        * data_RIGA
        * trainer_mode = 1
    * 1训练(vote=0)，验证使用7个（1-6+vote）（只训练head，gt=vote，不需要rater输入了）
        * data_RIGA_1raterin_1raterout
        * trainer_mode = 2
    * 每个rater单独prompt only（baseline）（只训练rater1-6）(head+prompt)
        * data_RIGA_new_1raterin_1raterout
        * trainer_mode = 3
3. Cls
    1. 6训练，6验证
    2. 1训练(vote=0)，验证使用6个（1-5+vote）（只训练head，gt=vote，不需要rater输入了）
    3. 每个rater单独prompt only（baseline）（只训练rater1-5）