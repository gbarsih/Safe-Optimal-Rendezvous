
clc
clear

r = 3;

v = ['[' sprintf('mu_%d,',1:r)];
v(end) = ']'
a = sym(v)