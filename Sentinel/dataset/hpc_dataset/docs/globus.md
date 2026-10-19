# How To install Globus

Ref: [https://docs.globus.org/globus-connect-personal/install/linux/](https://docs.globus.org/globus-connect-personal/install/linux/)



````
wget https://downloads.globus.org/globus-connect-personal/linux/stable/globusconnectpersonal-latest.tgz
tar xzf globusconnectpersonal-latest.tgz
# this will produce a versioned globusconnectpersonal directory
# replace `x.y.z` in the line below with the version number you see
cd globusconnectpersonal-x.y.z
# for the first time you are running it, you must complete setup
./globusconnectpersonal
````

# How to use Globus

````
# start globus
./globusconnectpersonal -start &
# stop globus
./globusconnectpersonal -stop
````

After starting globus, the Globus website would be able to see the machine running globus. You could download dataset on it.



# How to Configure Globus: Directory Permissions

open `~/.globusonline/lta/config-paths`, you would see a headerless CSV with fields defined as follows:

````
<path>,<sharing flag>,<R/W flag>
<path>,<sharing flag>,<R/W flag>
````

- Path: An absolute path to be permitted. Only paths which are present in the config file can be accessed. Tilde (~) can be used to represent the home directory of the user running Globus Connect Personal.
- Sharing Flag: Enable or disable sharing. This field must be 1 or 0. 1 allows sharing for the path and 0 disallows sharing.
- R/W Flag: Enable or disable write-access. This field must be 1 or 0. 1 allows read/write access and a 0 allows read-only access. The permissions set by this field are in addition to any other permissions and restrictions, e.g. file system permissions.

By default, read-write access to the user’s home directory is allowed, and sharing is disabled.

````
~/,0,1
````

Usually, You could add add directory permission for `path`:

````
~/,0,`
<path>,0,1
````

# Download Dataset

| Dataset Name | # of telemetry | # of hosts | size (GB) | data source                                       | data collect frequency                | Download |
| ------------ | -------------- | ---------  | --------- | ------------------------------------------------- | ------------------------------------- | -------- |
| OLCF         | 28             | 4626       | 492       | GPULog, OpenBMC, Job scheduler allocation history | 10 sec per host | [1,2]    |


[1] [Long Term Per-Component Power and Thermal Measurements of the OLCF Summit System](https://doi.ccs.ornl.gov/dataset/086578e9-8a9f-56b1-a657-0ed8b7393deb)  

[2] [OLCF Summit Supercomputer GPU Snapshots During Double-Bit Errors and Normal Operations](https://doi.ccs.ornl.gov/dataset/56c244d2-d273-5222-8f4b-f2324282fab8)


After downloading datasets, the file structure would be similar to:

```
rawdata/
├── 10.13139_OLCF_1861393/
│   ├── powtemp_10sec_mean/
│   │   ├── 202001/
│   │   │   └── *.parquet
│   │   ├── 202008/
│   │   │   └── *.parquet
│   │   ├── 202102/
│   │   │   └── *.parquet
│   │   ├── 202108/
│   │   │   └── *.parquet
│   │   └── 202201/
│   │       └── *.parquet
│   └── README.txt
└── 10.13139_OLCF_1970187/
    ├── *.csv
    └── README.md
```
