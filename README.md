# CrosswalkWatcher

## Initialize Project

### Initialize Submodule
<pre>
> git submodule update --init --recursive
</pre>

### Create Python Vertual Enviroment
<pre>
> python3 -m venv .env
> source .env/bin/activate
</pre>

### Install Yolo v5 requirements
<pre>
> cd (project)/external/yolo_v5_deepsort
> pip install -r requirements.txt
</pre>

### Detection
<pre>
> cd (project)/crosswalk_watcher
> python detect.py .. (options)
</pre>