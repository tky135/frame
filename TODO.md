Why .csv: to make sure previous split is kept when we test; split can be exported
A very dog thing about pandas: when it write strings, there's no "". So if a string are all digits, when it's read in again, it will be treated as an integer
A very dog thing about json: only allow string keys
##### TODO
1. Allow user to explicitly specify the test list
1. Make continue training better
1. make config a python file?
1. make all the writing local to this project? May not have right
1. testing and inferencing
1. Add result analysis for different tasks
1. Aim for better speed
