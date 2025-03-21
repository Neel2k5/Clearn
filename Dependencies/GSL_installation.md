# :gear: Installing GSL
---
### :pushpin: Debian

1. Update the package list
```
sudo apt update
```
2. Install GSL
```
sudo apt install libgsl-dev
```
3. Verify the installation
```
gsl-config --version
```
### :pushpin: Arch
1. Update the package list
```
sudo pacman -Syu
```
2. Install GSL
```
sudo pacman -S gsl
```
3. Verify the installation
```
gsl-config --version
```
### :pushpin: macOS
Using Homebrew
1. Update the package list
```
brew update
```
2. Install GSL
```
brew install gsl
```
3. Verify the installation
```
gsl-config --version
```
### :pushpin: Windows
- Using MSYS2 the process is the same as **Arch**
- Using WSL the process is the same as **Debian** 