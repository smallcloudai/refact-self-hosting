# Server Web UI getting started

After starting the server, you can access it in browser at
```commandline
http://localhost:8008
```

## Setting up models

By default server sets **CONTRASTcode/3b/multi** model on all available GPUs.

You can change it to another model or if you have more than one GPU, add models
with Chat/Toolbox features.


## Enable 3rd Party APIs

Server provides 3rd Party Toolbox and Chat powered by OpenAI models.
To use it you need **OpenAI API key**.

Set up and save your OpenAI API key in server settings (top right of the interface).
Then go to models tab and enable OpenAI API. If your IDE already runnning just relogin
to enable this features.