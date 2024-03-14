<!--SKIP_FIX-->
#### Tkinter for Python 3.11
1. Search the web for the correct `python3-tk.*deb` file.
   1. You can start here: https://mirrors.wikimedia.org/ubuntu/ubuntu/pool/main/p/python3-stdlib-extensions/
   2. In this folder, you can find .deb for Python 3.11.5.


2. Install this deb:
    ```
    sudo dpkg -i python3-tk_3.11.5-1_amd64.deb
    ```
    After that, you should find `_tkinter.cpython-311-x86_64-linux-gnu.so` in `/usr/lib/python3.11/lib-dynload`
    and a `tkinter` folder with .py files in `/usr/lib/python3.11`
    
    https://askubuntu.com/questions/1397737/how-to-install-tkinter-for-python-3-9-on-xubuntu-20-04

   If anything goes wrong, just copy those folders from the extracted package.

3. You may also need tk8.6-blt2.5:
    ```
    sudo apt-get install tk8.6-blt2.5
    ```