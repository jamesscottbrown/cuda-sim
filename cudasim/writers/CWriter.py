import os
from cudasim.relations import *
from cudasim.writers.Writer import Writer


class CWriter(Writer):
    def __init__(self, parsedModel, outputPath=""):
        Writer.__init__(self)
        self.parsedModel = parsedModel
        self.hppOutputFile = open(os.path.join(outputPath, self.parsedModel.name + ".hpp"), "w")
        self.cppOutputFile = open(os.path.join(outputPath, self.parsedModel.name + ".cpp"), "w")
        self.rename()

    def rename(self):
        """
        This function renames parts of self.parsedModel to meet the specific requirements of this writer.
        This behaviour replaces the previous approach of subclassing the parser to produce different results depending
        on the which writer was intended to be used.
        """

        # Remove any zero-padding from single-digit parameter names
        # This reverses any change applied by one of the CUDA writers
        for i in range(self.comp-1, len(self.parsedModel.parameterId)):
            old_name = self.parsedModel.parameterId[i]
            num = old_name[len('parameter'):]
            if len(num) > 1 and num[0] == '0':
                new_name = 'parameter' + str(num[1:])
                self.parsedModel.parameterId[i] = new_name
                self.parsedModel.rename_everywhere(old_name, new_name)

        # Remove any zero-padding from single-digit species names
        # This reverses any change applied by one of the CUDA writers
        for i in range(0, len(self.parsedModel.speciesId)):
            old_name = self.parsedModel.speciesId[i]
            num = old_name[len('species'):]
            if len(num) > 1 and num[0] == '0':
                new_name = 'species' + str(num[1:])
                self.parsedModel.speciesId[i] = new_name
                self.parsedModel.rename_everywhere(old_name, new_name)

    def write(self):
        self.write_c_header()
        self.write_c_source_code()

    def write_c_header(self):
        self.hppOutputFile.write("#ifndef ")
        self.hppOutputFile.write(self.parsedModel.name.upper())
        self.hppOutputFile.write("_HPP_\n")

        self.hppOutputFile.write("#define ")
        self.hppOutputFile.write(self.parsedModel.name.upper())
        self.hppOutputFile.write("_HPP_\n")

        self.hppOutputFile.write("""

        #include <vector>
        #include <iostream>
        #include "newmat.h"
        #include "newmatio.h"
        #include "newmatap.h"
        class ChildModel {
              public:

              /**
               * Number of reactions of the model
               */
              int NREACTIONS;

              /**
               * Number of species of the model
               */
              int NSPECIES;

              /**
               * Stoichiometric Matrix of the system (the rows represent the species and the columns the reactions)
               */
              Matrix* pstoichiometricMatrix;

              ChildModel(int i);
              void init();

             /**
               * Virtual method (ie method defined in the child class) setting the values of the stoichiometric matrix
               *
               * @param void
               * @return void
               */
              void getStoichiometricMatrix();

              /**
               * Virtual method computing the hazards of the different reactions for a given concentration of species (yi) and some parameter values
               *
               * @param double concentrations[] Array of size NSPECIES containing the concentrations of the species for which we want to compute the hazards
               * @param double parameters[] Array containing the parameter's values for which we want to compute the hazards (the number of parameters depend on the model and doesn't have to be the number of reactions)
               */
              ColumnVector getHazards(const double concentrations[],
                            const double parameters[]);

              /**
               * Virtual method modifying the concentrations and parameters depending on some criteria defined by the SBML
               *
               * @param double concentrations[] Array of size NSPECIES containing the concentrations of the species
               * @param double parameters[] Array containing the parameter's values
               */
              void applyRulesAndEvents(double concentrations[],
                            double parameters[], double time);
            """)

        for i in range(len(self.parsedModel.listOfFunctions)):

            self.hppOutputFile.write("double ")
            string = self.parsedModel.listOfFunctions[i].getId()
            string = re.sub('_', '', string)
            self.hppOutputFile.write(string)
            self.hppOutputFile.write("(")
            self.hppOutputFile.write("\tdouble\t" + self.parsedModel.functionArgument[i][0])

            for j in range(1, self.parsedModel.listOfFunctions[i].getNumArguments()):
                self.hppOutputFile.write(", ")
                self.hppOutputFile.write("double " + self.parsedModel.functionArgument[i][j])
            self.hppOutputFile.write(");\n")

        self.hppOutputFile.write('\n};\n')
        self.hppOutputFile.write('#endif /*')
        self.hppOutputFile.write(self.parsedModel.name.upper())
        self.hppOutputFile.write('_HPP_ */\n')

    def write_c_source_code(self):
        p1 = re.compile('species(\d+)')
        p2 = re.compile('parameter(\d+)')

        # self.cppOutputFile.write('#include "' + self.parsedModel.name + '.hpp"\n')
        self.cppOutputFile.write('#include "ChildModel.hpp"\n')
        self.cppOutputFile.write('#include <cmath>\n')
        self.write_model_constructor()
        self.write_user_defined_functions()
        self.write_stoichiometric_matrix()
        self.write_get_hazard_function(p1, p2)
        self.write_rules_and_events(p1, p2)

    def write_model_constructor(self):

        self.cppOutputFile.write("\nChildModel::ChildModel(int i){")
        self.cppOutputFile.write("\n\tNSPECIES = " + str(self.parsedModel.numSpecies) + ";")
        self.cppOutputFile.write("\n\tNREACTIONS = " + str(self.parsedModel.numReactions) + ";")
        self.cppOutputFile.write("\n\tpstoichiometricMatrix = new Matrix(NSPECIES,NREACTIONS);")
        self.cppOutputFile.write("\n\t(*pstoichiometricMatrix) = 0.0;")
        self.cppOutputFile.write("\n\tgetStoichiometricMatrix();")
        self.cppOutputFile.write("\n}")

    def write_user_defined_functions(self):

        # The user-defined functions used in the model must be written in the file

        for i in range(len(self.parsedModel.listOfFunctions)):
            self.cppOutputFile.write("double ChildModel::")
            string = self.parsedModel.listOfFunctions[i].getId()
            string = re.sub('_', '', string)
            self.cppOutputFile.write(string)
            self.cppOutputFile.write("(")

            self.cppOutputFile.write("double  " + self.parsedModel.functionArgument[i][0])
            for j in range(1, self.parsedModel.listOfFunctions[i].getNumArguments()):
                self.cppOutputFile.write(",")
                self.cppOutputFile.write(" double  " + self.parsedModel.functionArgument[i][j])
            self.cppOutputFile.write("){\n\n\t\tdouble output=")
            self.cppOutputFile.write(self.parsedModel.functionBody[i] + ";")
            self.cppOutputFile.write("\n\n\t\treturn output;\n\t}\n")

    def write_stoichiometric_matrix(self):

        self.cppOutputFile.write("\n\n\tvoid ChildModel::getStoichiometricMatrix() {")

        for i in range(self.parsedModel.numReactions):
            for k in range(self.parsedModel.numSpecies):
                ##if (self.parsedModel.species[k].getConstant() == False):
                self.cppOutputFile.write("\n\t\t (*pstoichiometricMatrix)(" + repr(k) + "+1," + repr(i) + "+1)= " + str(
                    self.parsedModel.stoichiometricMatrix[k][i]) + ";")
        self.cppOutputFile.write("\n\t}")

    def write_get_hazard_function(self, p1, p2):
        self.cppOutputFile.write(
            "\n\n\tColumnVector ChildModel::getHazards(const double concentrations[],const double parameters[]) {")
        self.cppOutputFile.write("\n\t\tColumnVector hazards(NREACTIONS);\n")
        for i in range(self.parsedModel.numReactions):
            string = self.parsedModel.kineticLaw[i]
            string = re.sub('_', '', string)
            string = p1.sub(r"concentrations[\g<1>-1]", string)
            string = p2.sub(r"parameters[\g<1>]", string)
            string = re.sub("compartment1", "parameters[0]", string)
            self.cppOutputFile.write("\n\t\thazards(" + repr(i) + "+1) = " + string)
            self.cppOutputFile.write(";\n")
        self.cppOutputFile.write("\t\treturn hazards;\n")
        self.cppOutputFile.write("\t}\n")

    def write_rules_and_events(self, p1, p2):

        # Write the rules and events
        self.cppOutputFile.write(
            "\n\tvoid ChildModel::applyRulesAndEvents(double concentrations[], double parameters[], double time) {\n")

        self.write_events(p1, p2)
        self.write_rules(p1, p2)

        self.cppOutputFile.write("\n\t}\n")

    def write_events(self, p1, p2):
        # Write the events

        for i in range(len(self.parsedModel.listOfEvents)):
            self.cppOutputFile.write("\t\tif ")
            string = mathMLConditionParser(self.parsedModel.eventCondition[i])
            string = re.sub(',', '>=', string)
            string = re.sub("geq", " ", string)
            self.cppOutputFile.write(string)
            self.cppOutputFile.write("{\n")
            list_of_assignment_rules = self.parsedModel.listOfEvents[i].getListOfEventAssignments()

            for j in range(len(list_of_assignment_rules)):
                self.cppOutputFile.write("\t\t\t")

                string = self.parsedModel.eventVariable[i][j]
                string = re.sub('_', '', string)
                string = p1.sub(r"concentrations[\g<1>-1]", string)
                string = p2.sub(r"parameters[\g<1>]", string)
                string = re.sub("compartment1", "parameters[0]", string)
                self.cppOutputFile.write(string)

                self.cppOutputFile.write("=")

                string = self.parsedModel.eventFormula[i][j]
                string = re.sub('_', '', string)
                string = p1.sub(r"concentrations[\g<1>-1]", string)
                string = p2.sub(r"parameters[\g<1>]", string)
                string = re.sub("compartment1", "parameters[0]", string)
                self.cppOutputFile.write(string)

                self.cppOutputFile.write(";\n\t\t}\n")
            self.cppOutputFile.write("\n")

        self.cppOutputFile.write("\n")

    def write_rules(self, p1, p2):
        # write the rules

        for i in range(len(self.parsedModel.listOfRules)):
            if self.parsedModel.listOfRules[i].isAssignment():
                self.cppOutputFile.write("\t\t")
                string = self.parsedModel.ruleVariable[i]
                string = re.sub('_', '', string)
                string = p1.sub(r"concentrations[\g<1>-1]", string)
                string = p2.sub(r"parameters[\g<1>]", string)
                string = re.sub("compartment1", "parameters[0]", string)
                self.cppOutputFile.write(string)

                self.cppOutputFile.write("=")

                string = mathMLConditionParser(self.parsedModel.ruleFormula[i])
                string = re.sub('_', '', string)
                string = p1.sub(r"concentrations[\g<1>-1]", string)
                string = p2.sub(r"parameters[\g<1>]", string)
                string = re.sub("compartment1", "parameters[0]", string)
                self.cppOutputFile.write(string)

                self.cppOutputFile.write(";\n")
