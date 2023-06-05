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
