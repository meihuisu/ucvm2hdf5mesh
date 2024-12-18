##
#  @file cvm_ucvm.py
#  @brief Common definitions and functions for UCVM plotting
#
#  Provides common definitions and functions that are shared amongst all UCVM
#  plotting scripts. Facilitates easier multi-processing as well.

#  Imports
from subprocess import call, Popen, PIPE, STDOUT
import sys
import os
import multiprocessing
import math
import struct
import getopt
import json

## Constant for all material properties.
ALL_PROPERTIES = ["vp", "vs", "density"]
## Constant for just Vs.
VS = ["vs"]
## Constant for just Vp.
VP = ["vp"]
## Constant for just density.
DENSITY = ["density"]

##
#  @class MaterialProperties
#  @brief Defines the possible material properties that @link UCVM UCVM @endlink can return.
#
#  Provides a class for defining the three current material properties that
#  UCVM returns and also has placeholders for Qp and Qs.
class MaterialProperties:
    
    ## 
    #  Initializes the MaterialProperties class.
    #  
    #  @param vp P-wave velocity in m/s. Must be a float.
    #  @param vs S-wave velocity in m/s. Must be a float.
    #  @param density Density in g/cm^3. Must be a float.
    #  @param poisson Poisson as a calculated float. Optional.
    #  @param qp Qp as a float. Optional.
    #  @param qs Qs as a float. Optional.
    def __init__(self, vp, vs, density, poisson = None, qp = None, qs = None):
       if pycvm_is_num(vp):
           ## P-wave velocity in m/s
           self.vp = float(vp)
       else:
           raise TypeError("Vp must be a number.")
       
       if pycvm_is_num(vs):
           ## S-wave velocity in m/s
           self.vs = float(vs)
       else:
           raise TypeError("Vs must be a number.")
       
       if pycvm_is_num(density):
           ## Density in g/cm^3
           self.density = float(density)
       else:
           raise TypeError("Density must be a number.")

       if poisson != None:
           self.poisson = float(poisson)
       else:
           self.poisson = -1
       
       if qp != None:
           ## Qp
           self.qp = float(qp)
       else:
           self.qp = -1
           
       if qs != None:
           ## Qs
           self.qs = float(qs)
       else:
           self.qs = -1
       
    ##
    #  Defines subtraction of two MaterialProperties classes.
    #
    #  @param own This MaterialProperties class.
    #  @param other The other MaterialProperties class.
    #  @return The subtracted properties.
    def __sub__(own, other):
        return MaterialProperties(own.vp - other.vp, own.vs - other.vs, own.density - other.density, \
                                  own.poisson - other.poisson, own.qp - other.qp, own.qs - other.qs)

    ##
    #  Initializes the class from a UCVM output string line.
    #
    #  @param cls Not used. Call as MaterialProperties.fromUCVMOutput(line).
    #  @param line The line containing the material properties as generated by ucvm_query.
    #  @return A constructed MaterialProperties class.
    @classmethod
    def fromUCVMOutput(cls, line):
        new_line = line.split()
        return cls(new_line[14], new_line[15], new_line[16])

    ##
    #  Initializes the class from a float list.
    #
    #  @param cls Not used. Call as MaterialProperties.fromNPFloats(flist).
    #  @param flist The flist line containing the material properties as imported from np array
    #  @return A constructed MaterialProperties class.
    @classmethod
    def fromNPFloats(cls, flist):
        vp=flist[0]
        vs=flist[1]
        density=flist[2]
        return cls(vp, vs, density)

    ##
    #  Initializes the class from a JSON  output string line.
    #
    #  @param cls Not used. Call as MaterialProperties.fromJSONOutput(jdict).
    #  @param jdict The jdict line containing the material properties as imported from file
    #  @return A constructed MaterialProperties class.
    @classmethod
    def fromJSONOutput(cls, jdict):
        vp=jdict['vp']
        vs=jdict['vs']
        density=jdict['density']
        return cls(vp, vs, density)

    ##
    #  Create a JSON output string line
    #
    #  @param depth The depth from surface.
    #  @return A JSON string
    def toJSON(self, depth):
        return "{ 'depth':%2f, 'vp':%.5f, 'vs':%.5f, 'density':%.5f }" % (depth, self.vp, self.vs, self.density)
    
    ##
    #  Retrieves the corresponding property given the property as a string.
    # 
    #  @param property The property name as a string ("vs", "vp", "density", "poisson", "qp", or "qs").
    #  @return The property value.
    def getProperty(self, property):               
        if property.lower() == "vs":
            return self.vs
        elif property.lower() == "vp":
            return self.vp
        elif property.lower() == "density":
            return self.density
        elif property.lower() == "poisson":
            return self.poisson
        elif property.lower() == "qp":
            return self.qp
        elif property.lower() == "qs":
            return self.qs
        else:
            raise ValueError("Parameter property must be a valid material property unit.")
    ##
    #  Set the corresponding property given the property as a string.
    # 
    #  @param property The property name as a string ("vs", "vp", "density", "qp", or "qs").
    #  @param val The property value.
    def setProperty(self, property, val):               
        if property.lower() == "vs":
            self.vs=val
        elif property.lower() == "vp":
            self.vp=val
        elif property.lower() == "density":
            self.density=val
        elif property.lower() == "poisson":
            self.poisson=val
        elif property.lower() == "qp":
            self.qp=val
        elif property.lower() == "qs":
            self.qs=val
        else:
            raise ValueError("Parameter property must be a valid material property unit.")
        
    ##
    #  String representation of the material properties.
    def __str__(self):
        return "Vp: %.2fm/s, Vs: %.2fm/s, Density: %.2fg/cm^3" % (self.vp, self.vs, self.density)
 
##
#  @class UCVM
#  @brief Python functions to interact with the underlying C code.
#
#  Provides a Python mechanism for calling the underlying C programs and
#  getting their output in a format that is readily and easily interpreted
#  by other classes.
class UCVM:
    
    ##
    #  Initializes the UCVM class and reads in all the available models that have
    #  been installed.
    #  
    #  @param install_dir The base installation directory of UCVM.
    #  @param config_file The location of the UCVM configuration file.
    def __init__(self, install_dir = None, config_file = None, z_range = None, floors = None):
        if install_dir != None:
            ## Location of the UCVM binary directory.
            self.binary_dir = install_dir + "/bin"
            self.utility_dir = install_dir + "/utilities"
        elif 'UCVM_INSTALL_PATH' in os.environ:
            mypath=os.environ.get('UCVM_INSTALL_PATH')
            self.binary_dir = mypath+"/bin"
            self.utility_dir = mypath+"/utilities"
        else:
            self.binary_dir = "../bin"
            self.utility_dir = "../utilities"
        
        if config_file != None:
            ## Location of the UCVM configuration file.
            self.config = config_file
        else:
            if install_dir != None:
               self.config = install_dir + "/conf/ucvm.conf"
            elif 'UCVM_INSTALL_PATH' in os.environ:
               mypath=os.environ.get('UCVM_INSTALL_PATH')
               self.config = mypath+"/conf/ucvm.conf"
            else:
               self.config = "../conf/ucvm.conf"

        if z_range != None:
            self.z_range = z_range
        else:
            self.z_range= None

        if floors != None:
            self.floors = floors
        else:
            self.floors= None
        
        
        if install_dir != None:
            ## List of all the installed CVMs.
            self.models = [x for x in os.listdir(install_dir + "/model")]
        elif 'UCVM_INSTALL_PATH' in os.environ:
            mypath=os.environ.get('UCVM_INSTALL_PATH')
            self.models = [x for x in os.listdir(mypath + "/model")]
        else:
            self.models = [x for x in os.listdir("../model")]
            
        self.models.remove("ucvm")

    ##
    #  Given raw UCVM result
    #   this function will throw an an error: missing model or invalid data etc
    #   by checking if first 'item' is float or not
    #
    #  @param raw An array of output material properties
    def checkUCVMoutput(self,idx,rawoutput):
        output = rawoutput.split("\n")[idx:-1]
        if len(output) > 1:
            line = output[0]
            if ("WARNING" in line) or ("slow performance" in line) or ("Using Geo" in line):
                return output
            p=line.split()[0]
            try :
                f=float(p)
            except :
                print("ERROR: "+str(line))
                exit(1)
           
        return output

    ##
    #  Queries UCVM given a set of points and a CVM to query. If the CVM does not exist,
    #  this function will throw an error. The set of points must be an array of the 
    #  @link Point Point @endlink class. This function returns an array of @link MaterialProperties
    #  MaterialProperties @endlink.
    #
    #  @param point_list An array of @link Point Points @endlink for which UCVM should query.
    #  @param cvm The CVM from which this data should be retrieved.
    #  @return An array of @link MaterialProperties @endlink.
    def query(self, point_list, cvm, elevation = None):
        shared_object = "../model/" + cvm + "/lib/lib" + cvm + ".so"
        properties = []
        
        # Can we load this library dynamically and bypass the C code entirely?
        if os.path.isfile(shared_object):
            import ctypes
            #obj = ctypes.cdll.LoadLibrary(shared_object)
            #print(obj)
        
        if( elevation ) :
            if self.z_range != None :
                if self.floors != None:
                  proc = Popen([self.utility_dir + "/run_ucvm_query.sh", "-f", self.config, "-m", cvm, "-c", "ge", "-z", self.z_range, "-L", self.floors], stdout=PIPE, stdin=PIPE, stderr=STDOUT, encoding='utf8')
                else:
                  proc = Popen([self.utility_dir + "/run_ucvm_query.sh", "-f", self.config, "-m", cvm, "-c", "ge", "-z", self.z_range], stdout=PIPE, stdin=PIPE, stderr=STDOUT, encoding='utf8')
            else :
                if self.floors != None:  ## z range is using default
                  proc = Popen([self.utility_dir + "/run_ucvm_query.sh", "-f", self.config, "-m", cvm, "-c", "ge", "-L", self.floors], stdout=PIPE, stdin=PIPE, stderr=STDOUT, encoding='utf8')
                else:
                  proc = Popen([self.utility_dir + "/run_ucvm_query.sh", "-f", self.config, "-m", cvm, "-c", "ge"], stdout=PIPE, stdin=PIPE, stderr=STDOUT, encoding='utf8')
        else :
            if self.z_range != None :
                if self.floors != None :
                  proc = Popen([self.utility_dir + "/run_ucvm_query.sh", "-f", self.config, "-m", cvm, "-c", "gd", "-z", self.z_range, "-L", self.floors], stdout=PIPE, stdin=PIPE, stderr=STDOUT, encoding='utf8')
                else:
                  proc = Popen([self.utility_dir + "/run_ucvm_query.sh", "-f", self.config, "-m", cvm, "-c", "gd", "-z", self.z_range], stdout=PIPE, stdin=PIPE, stderr=STDOUT, encoding='utf8')
            else:
                if self.floors != None:  ## z range is using default

                  proc = Popen([self.utility_dir + "/run_ucvm_query.sh", "-f", self.config, "-m", cvm, "-c", "gd", "-L", self.floors ], stdout=PIPE, stdin=PIPE, stderr=STDOUT, encoding='utf8')
                else:
                  proc = Popen([self.utility_dir + "/run_ucvm_query.sh", "-f", self.config, "-m", cvm, "-c", "gd" ], stdout=PIPE, stdin=PIPE, stderr=STDOUT, encoding='utf8')

        
        text_points = ""
        
        if isinstance(point_list, Point):
            point_list = [point_list]
         
        for point in point_list:
            if( elevation ) :
              text_points += "%.5f %.5f %.5f\n" % (point.longitude, point.latitude, point.elevation)
            else:
              text_points += "%.5f %.5f %.5f\n" % (point.longitude, point.latitude, point.depth)

        output = proc.communicate(input=text_points)[0]
        output = self.checkUCVMoutput(1,output)

        for line in output:
# it is material properties.. line
            try :
              mp = MaterialProperties.fromUCVMOutput(line)
              properties.append(mp)
            except :
              pass


        if len(properties) == 1:
            return properties[0]

        return properties

    ##
    #  Gets the Poisson value for a given set of Vs, Vp pair
    #  @param vs 
    #  @param vp
    #  @return poisson value
    def poisson(self, vs, vp) :
       if vs == 0 :
          return 0.0

       if vp == 0 :
          return 0.0

       return vp/vs

#  Function Definitions
##
#  Displays an error message and exits the program.
#  @param message The error message to be displayed.
def pycvm_display_error(message):
    print("An error has occurred while executing this script. The error was:\n")
    print(message)
    print("\nPlease contact software@scec.org and describe both the error and a bit")
    print("about the computer you are running CVM-S5 on (Linux, Mac, etc.).")
    exit(0)
    
##
#  Returns true if value is a number. False otherwise.
#
#  @param value The value to test if numeric or not.
#  @return True if it is a number, false if not.
def pycvm_is_num(value):
    try:
        float(value)
        return True
    except Exception:
        return False




##
#  @class Point
#  @brief Defines a point in WGS84 latitude, longitude projection.
#
#  Allows for a point to be defined within the 3D earth structure.
#  It has a longitude, latitude, and depth/elevation as minimum parameters,
#  but you can specify a type, e.g. "LA Basin", and a description, 
#  e.g. "New point of interest".
class Point:
    
    ##
    #  Initializes a new point. Checks that the parameters are all valid and
    #  raises an error if they are not.
    # 
    #  @param longitude Longitude provided as a float.
    #  @param latitude Latitude provided as a float.
    #  @param depth The depth in meters with the surface being 0.
    #  @param elevation The elevation in meters.
    #  @param type The type or classification of this location. Not required.
    #  @param code A short code for the site (unique identifier). Not required.
    #  @param description A longer description of what this point represents. Not required.
    def __init__(self, longitude, latitude, depth = 0, elevation = None, type = None, code = None, description = None):
        if pycvm_is_num(longitude):
            ## Longitude as a float in WGS84 projection.
            self.longitude = longitude
        else:
            raise TypeError("Longitude must be a number")
        
        if pycvm_is_num(latitude):
            ## Latitude as a float in WGS84 projection.
            self.latitude = latitude
        else:
            raise TypeError("Latitude must be a number")

        self.elevation = None;
        if elevation != None :
            self.elevation = elevation
        
        if pycvm_is_num(depth):
            if depth >= 0:
                ## Depth in meters below the surface. Must be greater than or equal to 0.
                self.depth = depth
            else:
                raise ValueError("Depth must be positive.")
        else:
            raise TypeError("Depth must be a number.")
        
        ## A classification or short description of what this point represents. Optional.
        self.type = type
        ## A short, unique code identifying the site. Optional.
        self.code = code
        ## A longer description of what this point represents. Optional.
        self.description = description
    
    ##
    #  String representation of the point.    
    def __str__(self):
        if(self.elevation) :
            return "(%.4f, %.4f, %.4f)" % (float(self.longitude), float(self.latitude), float(self.elevation))
        else:
            return "(%.4f, %.4f, %.4f)" % (float(self.longitude), float(self.latitude), float(self.depth))


