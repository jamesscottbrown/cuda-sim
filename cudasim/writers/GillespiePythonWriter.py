from libsbml import *
from cudasim.relations import *
import os
import sys
from Writer import Writer


class GillespiePythonWriter(Writer):
    def __init__(self, parser, output_path=""):
        Writer.__init__(self)
        self.parser = parser
        self.out_file = open(os.path.join(output_path, self.parser.parsedModel.name + ".py"), "w")
        self.rename()

    def rename(self):
        """
        This function renames parts of model to meet the specific requirements of this writer.
        This behaviour replaces the previous approach of subclassing the parser to produce different results depending
        on the which writer was intended to be used.
        """

        # Remove any zero-padding from single-digit parameter names
        # This reverses any change applied by one of the CUDA writers
        for i in range(self.parser.comp-1, len(self.parser.parsedModel.parameterId)):
            old_name = self.parser.parsedModel.parameterId[i]
            num = old_name[len('parameter'):]
            if len(num) > 1 and num[0] == '0':
                new_name = 'parameter' + str(num[1:])
                self.parser.parsedModel.parameterId[i] = new_name
                self.parser.rename_everywhere(old_name, new_name)

        # Remove any zero-padding from single-digit species names
        # This reverses any change applied by one of the CUDA writers
        for i in range(len(self.parser.parsedModel.speciesId)):
            old_name = self.parser.parsedModel.speciesId[i]
            num = old_name[len('species'):]
            if len(num) > 1 and num[0] == '0':
                new_name = 'species' + str(num[1:])
                self.parser.parsedModel.speciesId[i] = new_name
                self.parser.rename_everywhere(old_name, new_name)

    def write(self):

        for i in range(len(self.parser.parsedModel.listOfRules)):
            if self.parser.parsedModel.listOfRules[i].isRate():
                sys.exit("\n Model '" + self.parser.parsedModel.name + "' contains at least one rate rule, so " +
                         "cannot be simulated with the Gillespie algorithm! Please change the simmulation Type! \n")

        self.out_file.write("from cudasim.relations import *\n\n#Functions\n")
        
        self.write_functions()
        self.write_hazards_function()
        self.write_stoichiometry_functions()
        self.write_reaction_dictionary()
        
        self.out_file.write("#Rules and Events\n")
        
        self.write_rules_function()
        self.write_events_function()
        
        self.out_file.close()

    def write_functions(self):
        # Write one function per reaction, which will ajusst the state by adding the corresponding stoichiometric vector
        model = self.parser.parsedModel
        for i in range(len(model.listOfFunctions)):
            arg_string = ",".join(model.functionArgument[i])
            self.out_file.write("def %s (%s):\n\n" % (model.listOfFunctions[i].getId(), arg_string))
            self.out_file.write("\toutput = %s\n\n" % model.functionBody[i])
            self.out_file.write("\treturn output\n\n")

    def write_reaction_dictionary(self):
        # Writes a function that maps from reaction index to name of the corresponding stoichiometry function
        model = self.parser.parsedModel
        self.out_file.write("#Dictionary of reactions\ndef defaultfunc():\n\tpass\n\ndef Switch():\n\tswitch = {\n")
        for i in range(model.numReactions):
            self.out_file.write("\t\t" + repr(i) + " : Reaction" + repr(i) + ",\n")
        self.out_file.write("\t\t\"default\": defaultfunc\n\t\t}\n\treturn switch\n\n")

    def write_stoichiometry_functions(self):
        model = self.parser.parsedModel
        self.out_file.write("#Gillespie Reactions\n\n")
        for i in range(model.numReactions):
            self.out_file.write("def Reaction" + repr(i) + "((")
            for k in range(model.numSpecies):
                self.out_file.write(model.speciesId[k])
                self.out_file.write(",")

            self.out_file.write(")):\n\n")

            for k in range(model.numSpecies):
                self.out_file.write("\t%s_new= %s + (%s)\n" %
                                    (model.speciesId[k], model.speciesId[k], model.stoichiometricMatrix[k][i]))

            new_species = ",".join(map(lambda x: x + "_new", model.speciesId))
            self.out_file.write("\n\treturn(%s)\n\n" % new_species)

    def write_hazards_function(self):
        model = self.parser.parsedModel

        self.out_file.write("\n#Gillespie Hazards\n\n")
        self.out_file.write("def Hazards((%s),parameter):\n\n" % (",".join(model.speciesId)))

        for i in range(len(model.parameterId)):
            self.out_file.write("\t%s = parameter[%s]\n" % (model.parameterId[i], repr(i)))

        self.out_file.write("\n")
        hazard_list = []
        for i in range(model.numReactions):
            self.out_file.write("\tHazard_%s = %s\n" % (repr(i), model.kineticLaw[i]))
            hazard_list.append("Hazard_%s" % i)

        self.out_file.write("\n\treturn(%s)\n\n" % ",".join(hazard_list))

    def write_rules_function(self):
        model = self.parser.parsedModel

        self.out_file.write("def rules((%s),(%s),t):\n\n" % (",".join(model.speciesId), ",".join(model.parameterId)))

        for i in range(len(model.listOfRules)):
            if model.listOfRules[i].isAssignment():
                self.out_file.write("\t%s = %s\n" % (model.ruleVariable[i], model.ruleFormula[i]))

        self.out_file.write("\treturn((%s),(%s))\n\n" % (",".join(model.speciesId), ",".join(model.parameterId)))

    def write_events_function(self):
        model = self.parser.parsedModel

        self.out_file.write("def events((%s),(%s),t):\n\n" % (",".join(model.speciesId), ",".join(model.parameterId)))

        for i in range(len(model.listOfEvents)):
            self.out_file.write("\tif %s:\n" % mathml_condition_parser(model.eventCondition[i]))

            for j in range(len(model.listOfEvents[i].getListOfEventAssignments())):
                self.out_file.write("\t\t%s = model.eventFormula[i][j]\n" % (model.eventVariable[i][j]))
            self.out_file.write("\n\n")

        self.out_file.write("\treturn((%s),(%s))\n\n" % (",".join(model.speciesId), ",".join(model.parameterId)))
