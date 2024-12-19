# Bridging the Gap: Enhancing LLM Performance for Low-Resource African Languages with New Benchmarks, Fine-Tuning, and Cultural Adjustments

**Authors:**
**Tuka Alhanai** <tuka@ghamut.com>, **Adam Kasumovic** <adam.kasumovic@ghamut.com>, **Mohammad Ghassemi** <ghassemi@ghamut.com>, **Aven Zitzelberger** <aven.zitzelberger@ghamut.com>, **Jessica Lundin** <jessica.lundin@gatesfoundation.org>, **Guillaume Chabot-Couture** <Guillaume.Chabot-Couture@gatesfoundation.org>

This repository contains the benchmarks, results, and all the code required to reproduce the results, tables, and figures presented in our [paper](https://arxiv.org/abs/2412.12417).

<u>More specifically, this repository contains:</u>

1. **Translated Winogrande Benchmarks:** Human and machine translations of [Winogrande](https://github.com/allenai/winogrande) into 8 African languages: Shona, Igbo, Bambara, Amharic, Sepedi, Sesotho, Swtswana, and Tsonga (as well as [preexisting translations](https://github.com/InstituteforDiseaseModeling/winogrande-mmlu-clinical-za) into Afrikaans, Zulu, and Xhosa).
2. **Translated MMLU-Clinical Benchmarks:** Human and machine translations of the clinical sections "college medicine", "clinical knowledge", and "virology" of [MMLU](https://github.com/hendrycks/test) into 8 African languages: Shona, Igbo, Bambara, Amharic, Sepedi, Sesotho, Swtswana, and Tsonga (as well as [preexisting translations](https://github.com/InstituteforDiseaseModeling/winogrande-mmlu-clinical-za) into Afrikaans, Zulu, and Xhosa for the "clinical knowledge" and "college medicine" sections; we translated the "virology" section into Afrikaans, Zulu, and Xhosa as well in this release).
3. **Human Annotation of Winogrande:** Human annotations of the Winogrande dataset assessing translation quality and appropriateness in 11 African languages: Shona, Igbo, Bambara, Amharic, Sepedi, Sesotho, Swtswana, Tsonga, Afrikaans, Zulu, and Xhosa.
4. **Scripts to Reproduce Results:** Code used to regenerate the results, tables, and figures presented in the paper.

Note that Meta's [Belebele](https://github.com/facebookresearch/belebele) in English as well as Shona, Igbo, Bambara, Amharic, Sepedi, Sesotho, Swtswana, Tsonga, Afrikaans, Zulu, and Xhosa is also included since it was used as an African language evaluation benchmark for our experiments.

A [PDF-version of a slideshow presentation](Bridging_the_Gap_Low_Resource_Languages_Presentation.pdf) depicting our work is also provided in this repository.

Our translated datasets can also be accessed on HuggingFace at the following link: (URL pending)

## Contents

[`data/`](data): Contains data files used by scripts in the [`scripts/`](scripts) folder. This folder includes the translated evaluation benchmarks as well.

[`scripts/`](scripts): Contains Python scripts to generate the results, tables, and figures presented in the paper (when possible to re-create them from the data provided). Instructions on how to run the scripts can be found inside the folder.

[`results/`](results): Contains the output from the Python files in [`scripts/`](scripts), and has an identical folder tree structure to [`scripts/`](scripts).

[`utils/`](utils): Contains utility functions used by the scripts.

**Navigate to any of these folders for more detailed information about each in the form of additional READMEs.**

## Benchmarks Download
The evaluation benchmarks (MMLU-Clinical [called *mmlu_cm_ck_vir/*], Winogrande [called *winogrande_s/*], and Belebele [called *belebele/*]) in English and 11 African languages are available as a [ZIP file](data/evaluation_benchmarks_afr_release.zip). 

The ZIP file also includes machine translations of the benchmarks (see folders suffixed with *_mt*), as well as
their backtranslations (see folders suffixed with *_bt*). 

Within each benchmark directory, files are suffixed with [ISO Language Codes](https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes) (or no suffix for the original English) to denote the language the file is in (or was translated from for backtranslated versions). These language codes are used throughout the project to denote languages.

## Installation

[//]: # (### 1. Clone the repository:)

[//]: # (```shell)

[//]: # (git clone <repo-url-here>)

[//]: # (cd <repo-name-here>)

[//]: # (```)
**Note that if cloning this repository, be sure to run `git lfs install` first, using `apt-get update; apt-get install git-lfs` to install Git LFS if it isn't already. This is necessary to actually download the full data files needed to run many scripts. Cloning with `git clone https://<your GitHub PAT>@github.com/ghamut/BMGF_AAAI_Paper.git` should allow you to clone this private repository.** See [here](https://stackoverflow.com/questions/2505096/clone-a-private-repository-github) for more information about cloning a private repository.

### 1. Set up the Python environment:
Feel free to use your IDE to automatically do this.
```shell
python3 -m venv venv  # Set up virtual environment
source venv/bin/activate  # Activate virtual environment
pip install --upgrade pip  # Update pip
pip install -r requirements.txt  # Install packages
conda install -c conda-forge firefox geckodriver  # Run this command to get Bokeh working for creating one of the figures
```

### 2. Set up `config.py` file (gitignored by default) containing secrets:
Note that this step is not required if you do not want to run any scripts that incur monetary costs (e.g. calling OpenAI's Batch API, which is not free).

```shell
cp config_template.py config.py
```
You will need to edit the newly created `config.py`, filling in the string for your OpenAI API key:
```python
gpt_api_key = "your-OpenAI-GPT-API-key-here"
```
Your OpenAI API key can be found here:
[Creating/finding your API key](https://platform.openai.com/api-keys)

If you would like to run every script to reproduce the results that does not incur any monetary cost, simply run the following:
```shell
./run_everything.sh
```
The scripts will run in parallel but are not compute-heavy. The results can be viewed in [`results/`](results). If using version control,
only the figures should come up as "modified" due to how saving PDFs/SVGs works. The figures will be *visually* identical to their predecessors, though.

## Disclaimer
The code in this repository was developed by IDM, the Bill & Melinda Gates Foundation, and [Ghamut Corporation](https://ghamut.com/) to further research in Large Language Models (LLMs) for low-resource African languages by allowing them to be evaluated on question-answering and commonsense reasoning tasks, like those commonly available in English. We’ve made it publicly available under the MIT License to provide others with a better understanding of our research and an opportunity to build upon it for their own work. We make no representations that the code works as intended or that we will provide support, address issues that are found, or accept pull requests. You are welcome to create your own fork and modify the code to suit your own modeling needs as contemplated under the MIT License.


## Acknowledgments

This repository includes data derived from the following datasets, each subject to their respective licenses (copied from their respective GitHub repositories):

1. **MMLU Dataset**
   - GitHub Repository: [https://github.com/hendrycks/test](https://github.com/hendrycks/test)
   - License: [LICENSE-MMLU](./LICENSE-MMLU)
   - For more licensing details, see the license terms specified in the file.
   - Citation (see below):
        ```
        @article{hendryckstest2021,
          title={Measuring Massive Multitask Language Understanding},
          author={Dan Hendrycks and Collin Burns and Steven Basart and Andy Zou and Mantas Mazeika and Dawn Song and Jacob Steinhardt},
          journal={Proceedings of the International Conference on Learning Representations (ICLR)},
          year={2021}
        }
        
        @article{hendrycks2021ethics,
          title={Aligning AI With Shared Human Values},
          author={Dan Hendrycks and Collin Burns and Steven Basart and Andrew Critch and Jerry Li and Dawn Song and Jacob Steinhardt},
          journal={Proceedings of the International Conference on Learning Representations (ICLR)},
          year={2021}
        }
        ```
2. **Winogrande Dataset**
   - GitHub Repository: [https://github.com/allenai/winogrande](https://github.com/allenai/winogrande)
   - License: [LICENSE-Winogrande](./LICENSE-Winogrande)
   - For more licensing details, see the license terms specified in the file.
   - Citation (see below):
       ```
        @article{sakaguchi2019winogrande,
          title={WinoGrande: An Adversarial Winograd Schema Challenge at Scale},
          author={Sakaguchi, Keisuke and Bras, Ronan Le and Bhagavatula, Chandra and Choi, Yejin},
          journal={arXiv preprint arXiv:1907.10641},
          year={2019}
        }
        ```
3. **Belebele Dataset**
   - GitHub Repository: [https://github.com/facebookresearch/belebele](https://github.com/facebookresearch/belebele)
   - Licenses: [LICENSE-Belebele-CC-BY-NC4.0](LICENSE-Belebele-CC-BY-NC4.0) and [LICENSE-Belebele-CC-BY-SA4.0](LICENSE-Belebele-CC-BY-SA4.0)
   - For more licensing details, see the license terms specified in the files.
   - Citation (see below):
       ```
       @inproceedings{bandarkar-etal-2024-belebele,
         title = "The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants",
         author = "Bandarkar, Lucas  and
           Liang, Davis  and
           Muller, Benjamin  and
           Artetxe, Mikel  and
           Shukla, Satya Narayan  and
           Husa, Donald  and
           Goyal, Naman  and
           Krishnan, Abhinandan  and
           Zettlemoyer, Luke  and
           Khabsa, Madian",
         booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
         month = aug,
         year = "2024",
         address = "Bangkok, Thailand and virtual meeting",
         publisher = "Association for Computational Linguistics",
         url = "https://aclanthology.org/2024.acl-long.44",
         pages = "749--775",
       }
       ```

Please note that the licenses for the included datasets are separate from and may impose additional restrictions beyond the repository's [main license](LICENSE.md).

## Citation
If you find this repository useful, please consider citing it:
```
@article{,
  title={Bridging the Gap: Enhancing LLM Performance for Low-Resource African Languages with New Benchmarks, Fine-Tuning, and Cultural Adjustments},
  author={Tuka Alhanai and Adam Kasumovic and Mohammad Ghassemi and Aven Zitzelberger and Jessica Lundin and Guillaume Chabot-Couture},
  year={2024}
}
```
