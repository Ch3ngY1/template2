import argparse
import yaml
import os

class BaseConfig(object):

    def __init__(self):

        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--config', type=str, default='/data2/chengyi/Multi_Raters/Ours/config/configs/base.yaml')

        
    def initialize(self):
        args = self.parser.parse_args()

        with open('/data2/chengyi/Multi_Raters/Ours/config/configs/base.yaml') as f:
            config = yaml.safe_load(f)

        if args.config != '/data2/chengyi/Multi_Raters/Ours/config/configs/base.yaml':
            with open(args.config) as f:
                derived_config = yaml.safe_load(f)

            config = {**config, **derived_config}


        for key, value in config.items():
            setattr(args, key, value)

        return args


    def save_result(self, acc_list, loss_list, loss_refine_list):

        acc_save_path = os.path.join(self.args.save_folder, self.args.exp_name) + '/acc_{}_std_{}.txt'.format(acc_mean, acc_std)
        loss_save_path = os.path.join(self.args.save_folder, self.args.exp_name) + '/loss_{}_std_{}.txt'.format(loss_mean, loss_std)
        loss_refine_save_path = os.path.join(self.args.save_folder, self.args.exp_name) + '/loss2_{}_std_{}.txt'.format(loss_refine_mean, loss_refine_std)

        with open(acc_save_path, "w") as f:
            for c, ac in enumerate(acc_list):
                f.write('train_{}_acc is {}\n'.format(c, ac))
        f.close()

        with open(loss_save_path, "w") as f:
            for c, ac in enumerate(loss_list):
                f.write('train_{}_loss is {}\n'.format(c, ac))
        f.close()

        with open(loss_refine_save_path, "w") as f:
            for c, ac in enumerate(loss_refine_list):
                f.write('train_{}_loss is {}\n'.format(c, ac))
        f.close()


# 加载YAML文件
# if args.config:
#     with open(args.config) as f:
#         config = yaml.safe_load(f)
# else:
#     # 如果未指定YAML文件，则使用默认值
#     config = {
#         'arg1': 'default_value1',
#         'arg2': 123,
#         'arg3': 3.14
#     }

# # 将YAML配置加载到args中
# for key, value in config.items():
#     setattr(args, key, value)

# # 输出参数值
# print(args.arg1)
# print(args.arg2)
# print(args.arg3)
