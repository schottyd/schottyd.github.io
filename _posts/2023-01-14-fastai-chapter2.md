# Deploying fast.ai Chapter 2 code in early 2024

## Issues Issues Issues
I'm working through the fast.ai Practical Deep Learning for Coders [course](https://course.fast.ai/) and the accompanying [fastbook](https://github.com/fastai/fastbook) in late 2023/early 2024. My initial impressions of the course are really good - [Jeremy Howard](https://jeremy.fast.ai/) has a wealth of experience and a lovely way of explaining things. That said, getting set up and getting going in Chapter 1 was a bit of a challenge and then, in Chapter 2 I found myself in python dependency hell which took many hours to unpick - partly because of my own confusion about  the various python environment management approaches and partly because things moved on since the videos.  Jeremy tries to simplify things with a github repo called [fastsetup](https://github.com/fastai/fastsetup) - supposedly just run this and everything will auto-magically work but alas, that wasn't my experience. The book is also more up to date than the video (I think) but when things still didn't work that added to my confusion. This post describes the issues I came across, at least the ones that I remember and solved, and how I ended up fixing them - in the hope that it's useful to anyone following the course around about now and encountering the same issues. In some cases I did kinda find answers hidden in the fastai [forums](https://forums.fast.ai/t/lesson-2-official-topic/96033) but I never could get the AI summary of the topics that Jeremy mentioned in the video to work and digging through reams of posts from mid-2022 wasn't very fulfilling, especially when the issue was a python dependency that wasn't relevant back then. 

## What issues? 
Here's a short list of the problems that I had and I'll dig into each one explaining how I fixed it.
1. Jupyter Notebook or Jupyter Lab?  Jupyter version 3 or version 4?  Installing notebook extensions has changed and anyway, do Notebook extensions work in Lab and do I need them?  Why does everything work in Colab but not on my local machine?   How do I get Jupyter to recognise my environment - do I need to install the ipykernel to make this work?  Getting everything working locally was by far the biggest time-sink I faced.
2. Compounding all this,  how do you get ipywidgets to work with Voila? iPywidgets is now on version 8 but doesn't work with Voila whereas Colab uses 7.7 and apparently does. Trying to fix this involved finding some unresolvable `spacy` dependency related to the Python version. 
3. Running the app using `gradio` was also challenging for me - using exactly the same code from Tanishq's excellent [tutorial](https://www.tanishq.ai/blog/posts/2021-11-16-gradio-huggingface.html) didn't work and I spent an age figuring out why. 
4. Finally, getting it all working on Hugging Face was harder than it needed to be - mainly because they now need you to use SSH instead of user-name / password for the git bits. 
5. As a bonus, let's chat about the Jupyter debugger. 

## Getting the environment set up locally
Okay, let's talk about this. When you run Jupyter, you can run either `Jupyter notebook` or `Juputer lab`. Jupyter Notebook is supposedly a simpler, and maybe, older, user interface whereas as far as I can tell, Jupyter lab is what most people use today and the current version of Jupyter is 4.x. Jeremy seems to be running notebook, maybe under Jupyter version 2 or something - it's not entirely clear. But he then installs a bunch of extensions. Jupyter extensions make the user interface much more useful but of course, I ended up running Jupyter Lab and trying to install the extensions. This seems to be unnecessary because Lab already includes most of the useful functionality natively, and as far as I could tell, most of the extensions that Jeremy talks about don't even work on Lab. Also, back in Jupyter 2 days, extensions were installed by compiling them with various commands that web search and ChatGPT will still tell you about but have mostly been deprecated but you'll waste tons of time chasing it all down. 

So the next thing is that Jeremy recommends is using Mamba for environment management, and to use fastsetup to get things working quickly. I had all sorts of dependency issues which I won't go into - yours will be different - but eventually, I deleted my python environments, started again, and this is how I got it all working. Cutting to the chase, what I learnt from this is **don't try to find some configuration of old versions of the packages that works with the given code  ... just install current versions and fix the code.**  To this end, don't use fastsetup. It probably won't work for you and it's just as quick and easy to install the environment yourself and you'll learn more doing so. 

Now to what I did, setting up your environment - the lessons:

1. Install [MiniForge](https://github.com/conda-forge/miniforge).  Honestly, I am at my wits end with python environments. Its the least interesting thing one can possibly spend time on and the most painful to unpick. As Jeremy says, the one thing you absolutely don't want to do is mess with your system's Python environment. If you're on some flavour of Linux, most likely you've got Python 2.x and Python3.x and the system needs both.  If you ever try creating a symlink from python to python3 to avoid having to type python3 all the time, you'll quickly find out how bad things can get. If you mess it up enough, you may end up reinstalling your OS so just - don't!.  So why MamabaForge and not Mamba, Conda, Anaconda, Miniconda, venv or any of the others?  Here's a great [article](https://aseifert.com/p/python-environments/) explaining this - too boring to go into here but I'm all in on Mamba and this worked great. 
2. One nice thing about MambaForge is that it defaults to conda-forge so you don't have to keep typing `mamba -c conda-forge`. On this note, I will say to install as much as possible from conda-forge and only revert to fastchan (fastai's package repo) if you can't find the package elsewhere. This will help make sure you are installing the latest packages, not some old version. 
3. Next - make sure you install all the packages you need for fastai in your own environment, not the base environment.  I can't remember where I read this but the base environment is for Mamba, not for you. I did try installing all the fastai stuff in the base environment and it didn't work until I moved it all to my own environment. 
4. Use mamba to install `jupyterlab`, `nbdev`, `ipywidgets`, `voila` in your environment using mamba
5. Use mamba to install python 3.11.7 in your environment. One of the fastai packages using `spacy` and at the time of writing, it's incompatible with python 3.12. 
6. Next use mamba to install `fastai` and `fastbook` from the fastchan channel. 
7. Finally, use mamba to install `sentencepiece` from conda-forge.  I have no idea what or why - there were various warnings at some stage about versions which went away when the latest version was installed.

At this point, run Jupyter lab and start working through the labs. 

## Changing the code for ipywidgets 8
iPyWidgets seems to have gone through some decent changes between version 7.x and version 8.x. Full release history is [here](https://pypi.org/project/ipywidgets/#history).  If you've followed along so far, you're on current versions of everything, and on something like 8.1.1 of ipywidgets, when you try and create your simple bear classifier inference code, the only issue you're going to encounter  is with this this:
```
btn_upload.data[-1]
```

because `widgets.Fileupload()` has changed.  Simply change this to 
```
btn_upload.value[-1].content.tobytes()
```
and everything should be fine. 

## Changing the code for recent updates to Gradio
I'm on gradio version 4.13.0 (`mamba list | grep gradio` when your environment is active) and the gradio Interface class seems to have changed a fair bit since Tanish wrote his tutorial.  Here, ChatGPT helped me a lot so this was solved fairly quickly but here's a summary of what you need to do.  The following line is broken
```
Interface(fn=predict, inputs=gr.inputs.Image(shape=(512, 512)), outputs=gr.outputs.Label(num_top_classes=3)).launch(share=True)`
```

so replace it with the following. Also, there's something stylistic so you might choose to use line breaks to make the code more readable. 
```
interface = gr.Interface(
    fn=predict, 
    inputs=gr.Image(), 
    outputs=gr.Label(num_top_classes=3)
)
interface.launch(share=True)
```

Inputs and outputs are handled differently and the shape parameter has been deprecated. 
Also, just to mention, when you come to publish this on Hugging Face, make sure you get rid of `share=True`.  Oh, and in case you're unsure - yes you do need to import gradio. 
## Setting up SSH for hugging face - the less obvious bits
Last but not least, when you're faffing about getting git set up on your system and it comes time to push to the HF repo using `git push`, you may get the error that userid and password are incorrect, even though you could swear blind that they are correct. This seems to be because HF now only supports SSH for this and not userid / password. Git / HF has a great [tutorial](https://huggingface.co/docs/hub/security-git-ssh) for creating SSH keys and adding your public key to your settings, so follow that. To get `git push` working though, you're going to need to type this in the command line to tell your local git repo where the remote git repo is: 
```
git remote set-url origin git@hf.co:spaces/schottyd/fastai_c2_bear_classifier
```

replacing the part after "spaces" with your own space name, of course. 

# That's all folks
So that's it - hopefully this helps someone and if not, at least I got to check off the last task in the chapter and now I can happily move on to Chapter 3. 

Comments welcome.  
