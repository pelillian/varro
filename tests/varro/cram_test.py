import sys
def printer(frame, event, arg):
  if event in ['exception', 'opcode']:
      #print(arg[0])
      print(frame, event, arg)
  return printer

sys.settrace(printer)

print("Here")
import pytrellis

print("here")
