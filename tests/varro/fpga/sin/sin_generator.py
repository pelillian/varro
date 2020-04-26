from math import sin

result = ""

for i in range(4096): 
	binrep = "{0:b}".format(i)
	if len(binrep) < 12: 
		for i in range(12 - len(binrep)): 
			binrep = "0{}".format(binrep)
	binrep = "{0}{1}".format("12'b", binrep)
	result = "{0}{1} : analog = {2};\n".format(result, binrep, abs(int(sin(i) * 63))) # "            " + binrep + " : analog = " + "{};\n".format(abs(int(sin(i) * 63)

print(result)
