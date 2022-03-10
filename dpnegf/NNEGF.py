import os
import re
import ase
import ase.io
import argparse

from dpnegf.Parameters import Paras2  as Paras
from dpnegf.nnet.Model import Model
from dpnegf.negf.NEGFStruct import StructNEGFBuild
from dpnegf.negf.SurfaceGF import SurfGF
from dpnegf.negf.NEGFHamilton import NEGFHamiltonian, Device_Hamils, Contact_Hamils
from dpnegf.negf.NEGF import NEGFcal

def deepnegf(args:argparse.Namespace):
    # command line. 
    #parser = argparse.ArgumentParser(description="Parameters.")  
    #parser.add_argument('-i', '--input_file', type=str,
    #                    default='inputnn.json', help='json file for inputs, default inputnn.json')
    #parser.add_argument('-s', '--struct', type=str, default='struct.xyz',
    #                    help='struct file name default struct.xyz')
    #parser.add_argument('-fmt', '--format', type=str, default='xyz',
    #                    help='struct file format default xyz')

    #args = parser.parse_args()

    input_file = args.input_file
    fp = open(input_file)
    paras = Paras(fp,args)

    structfile = args.struct
    structfmt = args.format

    structase = ase.io.read(structfile,format=structfmt)
    negfH = NEGFHamiltonian(paras,structase)
    ScatDict, ScatContDict = negfH.Scat_Hamils()
    ContDict = negfH.Cont_Hamils()

    negfcal = NEGFcal(paras)
    negfcal.Scat_Hamiltons(HamilDict = ScatDict,Efermi = paras.DeviceFermi)
    negfcal.Scat_Cont_Hamiltons(HamilDict = ScatContDict,Efermi = paras.DeviceFermi)
    negfcal.Cont_Hamiltons(HamilDict = ContDict,Efermi = paras.ContactFermi)
    negfcal.Cal_NEGF()


#if __name__ == "__main__":
#    deepnegf()