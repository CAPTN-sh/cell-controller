A real-time controller for MQTT cellular modem data, performing AI-based (Deep Autoencoders) anomaly detection, trends in 4/5G parameters, cell ID/network switch detection, and alerting via MQTT to the application layer controller. Rely on the raw data provided by ADDIX Wavelab MQTT brokers in the scope of the CAPTN-projects.

# Installation
## Requirements
- poetry
- make
- (optional) mosquitto [if a local MQTT broker is needed]

## Dependencies
install the project deps within the poetry environment:
```bash
make install
```
or globally
```bash
make install-global
```

# Demo
At the time of writing **(16 December 2024)** there is a running Docker container on the CAU server at the Wavelab (`192.168.237.62`, behind the Wavelab VPN). The container is called `gstreamer-gstwebrtcapp-cuda-1` and provides all the necessary infrastructure an dependencies to run it. This folder is available in the `/home/cell_controller` directory inside the container.

# Deployment
The project could be deployed anywhere where the MQTT broker with the raw data is accessible for subscription. It does not require docker/k8 or any special hardware. The local MQTT broker via `mosquitto` could be used for test publishing or could be ignored at all by using the verbose mode (stdout) for the controller and omitting the publishing MQTT broker configuration.

# Usage
Run the controller with a provided entrypoint script and a configuration file:
```bash
python run.py -c <path_to_config_file>
```

The exemplary configuration file for the Wavelab broker is provided under `examples` folder. There are also the explanatory comments for each configuration parameter.

> [!NOTE]
> You must request the credentials from ADDIX to subscribe to the Wavelab MQTT brokers and fill the credentials in the configuration file. 
> Until that the controller will throw a runtime error due to a failed connection.

In the container, there is a `config.yaml` file in the root project folder with the credentials for the Wavelab MQTT broker. You need first to attach a shell to the container after being logged in the CAU server via SSH:
```bash
docker exec -it gstreamer-gstwebrtcapp-cuda-1 bash
```

Then you can run the service with the following command:
```bash
cd /home/cell_controller && python run.py -c config.yaml
```
To detach from the container, press `Ctrl+P` and `Ctrl+Q` in the attached shell in sequence. To run the service in the non-blocking mode, you can use the `disown` command:
```bash
cd /home/cell_controller && python run.py -c config.yaml & disown
```
The logging could be disabled by setting the `verbose` parameter to `false` in the configuration file.

# License
This project is licensed under the CC BY-NC 4.0 License - see the [LICENSE](LICENSE) file for details.

# Acknowledgments
The development of this software has been funded by the German Federal Ministry for Digital and Transport (Bundesministerium für Digitales und Verkehr) within the project "CAPTN Förde 5G", funding guideline: "5G Umsetzungsförderung im Rahmen des 5G-Innovationsprogramms", funding code: 45FGU139_H. The authors acknowledge the financial support of the BMDV.

# Authors
M.Sc. Nikita Smirnov, Prof. Dr.-Ing. Sven Tomforde, Intelligent Systems AG, Department of Computer Science, Kiel University, Germany.

Please open an issue or contact me in case of any questions or bug reports: [mailto Nikita Smirnov](mailto:nikita.smirnov@cs.uni-kiel.de)
