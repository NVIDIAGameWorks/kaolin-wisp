# Wisp Applications

<img src="../media/wisp_main.jpg" alt="Main Script" width="750"/>

The `app` folder contains various apps built with Wisp building blocks.

Each app includes its own set of argument definitions, configs, README file and where needed, also a `requirements.txt` for installing additional packages.

Users are welcome to extend & modify these apps, as well as use them as a guidance for using Wisp.

## External Contributions

Contributions are welcome!

Have you built a useful app with Wisp? Message us or leave a PR:
* Apps with their own dedicated repo will be linked from the main page.
* Alternatively, you're welcome to submit a PR to add your app under this folder. 

## Interactive Renderer v.s. HEADLESS mode
Where possible, Wisp apps support both interactive and headless mode by modifying the `$WISP_HEADLESS` env variable
(where `$WISP_HEADLESS=1` turns off the interactive renderer).

Note the synergy of the various components:
* Apps should always implement a `main_x.py` script.
* Interactive apps are likely to generate an instance of a `WispApp` subclass.
* `WispApp` can communicate with Trainers and other components via the `WispState` object. This
maintains a clear separation of the gui logic and core code.
