# Credit Card Application Fraud Analytics <br/> 信用卡欺诈分析

Stephen Coggeshall 金融欺诈线上课程第二组（2020-04 -- 2020-05）

## Roles

- Feature Selection
    - Filter
        - 姚舜, 文铁
    - Wrapper
        - 韩耀东, 田震宇

- Model Parameter Tuning
    - Logistic regression
        - params: C=1, class_weight='balanced'
        - FDR: 50.05%
    - Random forest
    - Neural network
    - Boosted tree
    - SVM
    - Decision tree
    - K nearest neighbor
    - Naive bayes

- Research and Report Writing
    
- GitHub repo
    - 韩耀东

## GitHub Workflow（工作流程）

### Notes

- **请大家一定不要直接修改主分支 (master)**

- GitHub repo 地址：https://github.com/Fraud-Analytics/project <br/>
工具 repo 地址：https://github.com/Fraud-Analytics/Useful_Tool

- 所有提交和分支在同一 GitHub repo 中管理，每个人的开发文件请放在 `dev` 目录下的个人文件夹内。
例如：`dev/Han_Yaodong`。（希望大家不要随意修改他人的文件😂）

- master 分支作为主分支，里面的代码日后直接作为写 report 的数据支持，所以合并到 master
分支时请大家一定仔细检查。

- 每个人分别开发时，建立自己的分支 (branch)，例如 `yaodong`。这样可以避免不同人的代码互相干扰。

- 合并代码到 master 时，如果提交的是 jupyter notebook 文件，请将所有代码块从头到尾**全部跑完**之后，
将**代码和结果**一起提交。这样可以保证在 master 分支上看到的输出值可以作为最终结果，和 report 的依据。

### 提交代码流程

1. 在个人文件夹和个人分支上开发代码，可以随意实验和建立临时文件，但是不要在 master 分支上修改，
也最好不要修改根目录里的代码。

1. 完成一部分代码的开发后，如果你想把你的代码分享给大家讨论或者参考，可以将修改的代码文件放在**个人文件夹**
里，push 到**个人分支**上。同理，如果想阅读别人的代码，可以 checkout 别人的分支。为了代码版本的整齐，
和后期数据的一致性，请不要直接在 master 分支上修改。

1. 如果你完成了你负责的代码，想要合并到 master 分支，请先建立一个新的分支（名字可以自取，最好能反映提交代码的功能，
例如：`update-readme`），将想要合并的代码从个人文件夹内拷贝到主目录下，并且提交到新的分支上。<br/>
（例如：将 `dev/Han_Yaodong/feature_selection.py` 拷贝到 `./feature_selection.py`）

1. 提交完成后，可以在微信群里告诉大家你的新分支。如果有时间，大家可以帮忙检查一下提交的文件是否有问题
（格式是否正确、代码块是否都有结果、是否包含了不必要的文件）。如果大家会使用 Pull Request 功能，尽量使用
Pull Request 提交合并请求。

1. 一切检查没问题后，再将新提交的分支合并到 master。Master 分支里的代码和结果数据应该时刻保持正确，
并且可以作为写 report 的根据。

## Plans

## Misc

