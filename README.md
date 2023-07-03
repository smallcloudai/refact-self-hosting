<div align="center">

# <font color="red">[{</font> Refact.ai Inference Server

</div>

This is a self-hosted server for the [refact.ai](https://www.refact.ai) coding assistant.

With Refact you can run high-quality AI code completions on-premise and use a number of
functions for code transformation and ask questions in the chat.

This server allows you to run AI coding models on your hardware, your code doesn't go outside your control.

At the moment, you can choose between following models:

| Model                                                                                | GPU (VRAM) | Completion | AI Toolbox | Chat | Fine tuning | Languages supported                                |
|--------------------------------------------------------------------------------------|-----------:|:----------:|:----------:|:----:|:-----------:| -------------------------------------------------- |
| [CONTRASTcode/medium/multi](https://huggingface.co/smallcloudai/codify_medium_multi) |        3Gb |     +      |            |      |             | [20+ Programming Languages](https://refact.ai/faq) |
| [CONTRASTcode/3b/multi](https://huggingface.co/smallcloudai/codify_3b_multi)         |        8Gb |     +      |            |      |      +      | [20+ Programming Languages](https://refact.ai/faq) |
| [starchat/15b/beta8bit](https://huggingface.co/rahuldshetty/starchat-beta-8bit)      |       16Gb |            |            |  +   |             | [80+ Programming languages](https://huggingface.co/blog/starchat-alpha) |
| [starcoder/15b/base4bit](https://huggingface.co/smallcloudai/starcoder_15b_4bit)     |       16Gb |     +      |     +      |  +   |             | [80+ Programming languages](https://huggingface.co/blog/starcoder) |
| [starcoder/15b/base8bit](https://huggingface.co/smallcloudai/starcoder_15b_8bit)     |       32Gb |     +      |     +      |  +   |             | [80+ Programming languages](https://huggingface.co/blog/starcoder) |

Refact is currently available as a plugin for [JetBrains](https://plugins.jetbrains.com/plugin/20647-refact-ai)
products and [VS Code IDE](https://marketplace.visualstudio.com/items?itemName=smallcloud.codify).



## Known limitations

- For best results on smaller GPUs we recommend using CONTRASTcode models as the StarCoder and StarChat models can be quite slow



## Demo

<table align="center">
<tr>
<th><img src="https://plugins.jetbrains.com/files/20647/screenshot_277b57c5-2104-4ca8-9efc-1a63b8cb330f" align="center"/></th>
</tr>
</table>



## Getting started

Install plugin for your IDE:
[JetBrains](https://plugins.jetbrains.com/plugin/20647-refact-ai) or
[VSCode](https://marketplace.visualstudio.com/items?itemName=smallcloud.codify).


### Running Server in Docker

The recommended way to run server is a pre-build Docker image.

#### Linux

Install Docker with NVidia GPU support following [this guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).

#### Windows

Install WSL 2 and then install Docker, [one guide to do this](https://docs.docker.com/desktop/install/windows-install).
After installation, run docker-desktop to up down docker service.


<details><summary>Docker tips & tricks</summary>

Add your yourself to docker group to run docker without sudo (works for Linux):
```commandline
sudo usermod -aG docker {your user}
```
List all containers:
```commandline
docker ps -a
```
Create a new container:
```commandline
docker run
```
Start and stop existing containers (stop doesn't remove them):
```commandline
docker start
docker stop
```
Remove a container and all its data:
```commandline
docker rm
```

Shows messages from the container:
```commandline
docker logs -f
```
</details>

Run docker container with following command:
```commandline
docker run --rm --gpus all -p 8008:8008 -v refact_workdir:/perm_storage smallcloud/refact_self_hosting_enterprise
```
After start container will automatically download **CONTRASTcode/3b/model**. It takes time and depends
on your internet connection.


### Running Manually

Coming soon...


## Server configuration

Server works with default parameters out-of-the-box. But most likely you will want to
configure it according your goals. Here is a [guide](docs/getting_started.md) to do it.


## Setting Up Plugins

Go to plugin settings and set up a custom inference url:
```commandline
http://localhost:8008
```
<details><summary>JetBrains</summary>
Settings > Tools > Refact.ai > Advanced > Inference URL
</details>
<details><summary>VSCode</summary>
Extensions > Refact.ai Assistant > Settings > Infurl
</details>


Now it should work, just try to write some code! If it doesn't, please report your experience to
[GitHub issues](https://github.com/smallcloudai/refact-self-hosting/issues).



## Fine Tuning

*Why?*  Code models are trained on a vast amount of code from the internet, which may not perfectly
align with your specific codebase, APIs, objects, or coding style.
By fine-tuning the model, you can make it more familiar with your codebase and coding patterns.
This allows the model to better understand your specific needs and provide more relevant and
accurate code suggestions. Fine-tuning essentially helps the model memorize the patterns and
structures commonly found in your code, resulting in improved suggestions tailored to your
coding style and requirements.

Read this [guide](docs/getting_started.md) to learn how to fine-tune your model.


## Community & Support

Join our
[Discord server](https://www.smallcloud.ai/discord) and follow our
[Twitter](https://twitter.com/refact_ai) to get the latest updates.



## Contributing

We are open for contributions. If you have any ideas and ready to implement this, just:
- make a [fork](https://github.com/smallcloudai/refact-self-hosting/fork)
- make your changes, commit to your fork
- and open a PR
