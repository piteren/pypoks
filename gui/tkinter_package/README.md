#### Tkinter for python 3.11
1. Search the web for the right `python3-tk.*deb` file. 
   1. You can start here: https://mirrors.wikimedia.org/ubuntu/ubuntu/pool/main/p/python3-stdlib-extensions/
   2. In this folder you can find .deb for python3.11.5. 
2. Install this deb: 
    ```
    sudo dpkg -i python3-tk_3.11.5-1_amd64.deb
    ```
    After that you should find `_tkinter.cpython-311-x86_64-linux-gnu.so` in `/usr/lib/python3.11/lib-dynload`
    and `tkinter` folder with .py files in `/usr/lib/python3.11`
    
    https://askubuntu.com/questions/1397737/how-to-install-tkinter-for-python-3-9-on-xubuntu-20-04