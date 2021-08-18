#!/home/rdkibler/.conda/envs/domesticator/bin/python3
# -*- coding: utf-8 -*-

##also look into pytest
import sys

database='/home/rdkibler/projects/domesticator/database/'
sys.path.insert(0, database)

#import json
from collections import Counter
#from dnachisel import *
from dnachisel import EnforceTranslation, AvoidChanges, DnaOptimizationProblem, CodonOptimize, AvoidPattern, HomopolymerPattern, EnzymeSitePattern, EnforceGCContent, EnforceTerminalGCContent, AvoidHairpins, NoSolutionError
from dnachisel import Location
from dnachisel import reverse_translate
import argparse
import warnings
from Bio import SeqIO
#from Bio.SeqFeature import *
from Bio.Seq import MutableSeq, Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import FeatureLocation, SeqFeature
from Bio.Alphabet import IUPAC
#from Bio.PDB import PDBParser, PPBuilder
from constraints import ConstrainCAI
from objectives import MinimizeSecondaryStructure, MinimizeKmerScore
import os
import copy

from dnachisel.DnaOptimizationProblem import DEFAULT_SPECIFICATIONS_DICT


CUSTOM_SPECIFICATIONS_DICT = copy.deepcopy(DEFAULT_SPECIFICATIONS_DICT)
CUSTOM_SPECIFICATIONS_DICT.update({
    'ConstrainCAI': ConstrainCAI,
    'MinimizeSecondaryStructure': MinimizeSecondaryStructure,
    'MinimizeKmerScore': MinimizeKmerScore
})

def load_template(filename, insert, destination):
	''' func descriptor '''

	objectives = []
	constraints = []

	vector = SeqIO.read(filename, "genbank")
	

	vector, insert_location = insert_into_vector(vector, destination, insert)

	problem = DnaOptimizationProblem.from_record(vector, specifications_dict=CUSTOM_SPECIFICATIONS_DICT)
	constraints += problem.constraints
	objectives += problem.objectives

	#feats = [feat.qualifiers for feat in vector.features]
	#dnachisel hasn't implemented MultiLocation yet
	#vector_location = FeatureLocation(insert_location.end, len(vector)) + FeatureLocation(0,insert_location.start)
	#vector_location_us = Location(0, insert_location.start, 1)
	#vector_location_ds = Location(insert_location.end, len(vector), 1)

	#constraints.append(EnforceTranslation(Location.from_biopython_location(insert_location)))
	#constraints.append(AvoidChanges(vector_location_us))
	#constraints.append(AvoidChanges(vector_location_ds))


	#This seq should be a SeqRecord object
	return vector, objectives, constraints, insert_location

def replace_sequence_in_record(record, location, new_seq):
	#print(record, location, new_seq)
	#print(dir(location))
	#print(location.extract(record.seq))
	#print(record.seq[location.start:location.end])
	
	if location.strand >= 0:
		adjusted_seq = record.seq[:location.start] + new_seq.seq + record.seq[location.end:]
	else:
		adjusted_seq = record.seq[:location.start] + new_seq.reverse_complement().seq + record.seq[location.end:]

	#exit(adjusted_seq)
	record.seq = adjusted_seq

	#print(help(location))
	#exit(dir(location))

	seq_diff = len(new_seq) - len(location)
	orig_start = location.start
	orig_end = location.end

	processed_features = []

	#print("-=-=-=-=-=-=-=-=-=-==---=-=-=-=-=-=")
	#print(location)
	#print("diff: %d" % seq_diff)

	#adjust all features
	for feat in record.features:
		#print("----------------------------------------")
		#print(feat.qualifiers['label'][0])
		#print(feat.location)

		f_loc = feat.location

		loc_list = []

		for subloc in f_loc.parts:

			assert(subloc.start <= subloc.end)
			
			#type 1: where the start and end are contained within the original location
			#-> do not add it to the processed_features list
			if subloc.start > location.start and subloc.start < location.end and subloc.end > location.start and subloc.end < location.end:
				#print("start: %d and end: %d are contained within %s" % (subloc.start, subloc.end, location))
				#print("omit")
				continue

			#type 1b: where the start and end are the same which will happen a lot for storing constraints and objectives
			elif subloc.start == location.start and subloc.end == location.end:
				new_loc = FeatureLocation(location.start, location.end + seq_diff, strand=subloc.strand)



			#type 2: where they start or end inside the location
			#-> chop off. don't forget to add on approprate amount
			##THINK! does strand even matter? How is start and end defined? I'm assuming that for strand -1 things are flipped but that's probably not how it's implemented. Also, consider strand = 0 (no direction). There is probably an easier way. 
			elif subloc.start >= location.start and subloc.start <= location.end:
				#print("start: %d is in %s" % (subloc.start, location))
				new_loc = FeatureLocation(location.end + seq_diff, subloc.end + seq_diff, strand=subloc.strand)

			elif subloc.end >= location.start and subloc.end <= location.end:
				#print("end: %d is in %s" % (subloc.end, location))
				new_loc = FeatureLocation(subloc.start, location.start, strand=subloc.strand)
				

			#type 3: where they span the location 
			#-> keep the leftmost point same and add diff to rightmost. do not split
			elif location.start >= subloc.start and location.start <= subloc.end and location.end >= subloc.start and location.end <= subloc.end:
				#print("loc spans insert. keep start and add diff to end")
				new_loc = FeatureLocation(subloc.start, subloc.end + seq_diff, strand=subloc.strand)

			#type 4: where they start and end before location
			#-> add it to list unchanged
			elif subloc.start <= location.start and subloc.end <= location.start:
				#print("loc is before insert location so just keep")
				new_loc = subloc

			#type 5: where they start and end after location
			#-> add diff to whole location
			elif subloc.start >= location.end and subloc.end >= location.end:
				#print("loc is after insert location so apply offset and keep")
				new_loc = subloc + seq_diff

			loc_list.append(new_loc)
			#print("new loc:")
			#print(new_loc)
		
		#if the list is empty, it means that all the sublocs were contained within the insert
		if loc_list:
			feat.location = sum(loc_list)
			processed_features.append(feat)


	record.features = processed_features


	return record

def load_user_options(args, f_location):

	#assert(isinstance(f_location, FeatureLocation))

	#I need location variable to be the dna_chisel version
	if isinstance(f_location, FeatureLocation):
		location = Location.from_biopython_location(f_location) #to dna_chisel
	else:
		location = f_location

	#set enforce translation to the whole thing
	constraints = []
	objectives = []

	if args.harmonized:
		opt_mode = 'harmonized'
	else:
		opt_mode = 'best_codon'
	objectives += [
		CodonOptimize(species=args.species, location=location, mode=opt_mode)
	]
	constraints += [
		EnforceTranslation(location=location)
	]

	if args.avoid_homopolymers:
		constraints += [
		AvoidPattern(HomopolymerPattern("A",args.avoid_homopolymers),location=location),
		AvoidPattern(HomopolymerPattern("T",args.avoid_homopolymers),location=location),
		AvoidPattern(HomopolymerPattern("G",args.avoid_homopolymers),location=location),
		AvoidPattern(HomopolymerPattern("C",args.avoid_homopolymers),location=location)]

	if args.avoid_hairpins:
		constraints += [AvoidHairpins(location=location)]

	if args.avoid_patterns:
		constraints += [AvoidPattern(pattern,location=location) for pattern in args.avoid_patterns]

	#NOTE! Printing this to a template is broken
	if args.avoid_restriction_sites:
		constraints += [AvoidPattern(EnzymeSitePattern(enzy),location=location) for enzy in args.avoid_restriction_sites]

	if args.constrain_global_GC_content:
		constraints += [EnforceGCContent(mini=args.global_GC_content_min, maxi=args.global_GC_content_max, location=location)]

	if args.constrain_local_GC_content:
		constraints += [EnforceGCContent(mini=args.local_GC_content_min, maxi=args.global_GC_content_max, window=args.local_GC_content_window, location=location)]

	if args.constrain_terminal_GC_content:
		constraints += [EnforceTerminalGCContent(mini=args.terminal_GC_content_min, maxi=args.terminal_GC_content_max, window_size=8, location=location)]

	if args.constrain_CAI:
		constraints += [ConstrainCAI(species=args.species, minimum=args.constrain_CAI_minimum, location=location)]

	if args.optimize_dicodon_frequency:
		objectives += [MaximizeDicodonAdaptiveIndex()]

	if args.kmers:
		objectives += [MinimizeKmerScore(k=args.kmers, boost=args.avoid_kmers_boost, location=location)]

	if args.avoid_secondary_structure:
		objectives += [MinimizeSecondaryStructure(max_energy=args.avoid_secondary_structure_max_e, location=location, boost=args.avoid_secondary_structure_boost)]

	if args.avoid_initiator_secondary_structure:
		objectives += [MinimizeSecondaryStructure(max_energy=args.avoid_initiator_secondary_structure_max_e, location=location, optimize_initiator=True, boost=args.avoid_initiator_secondary_structure_boost)]

	return objectives, constraints

def find_annotation(record, label):
	for feat in record.features:
		if label == feat.qualifiers['label'][0]:
			#I will be replacing it so remove it:
			#vector.features.remove(feat)
			return feat
	exit("label not found: " + label)

def insert_into_vector(vector, destination, new_seq):
	
	destination_annotation = find_annotation(vector, destination)
	#print(destination_annotation)

	location = destination_annotation.location

	#print(vector)
	#print(vector.features)
	#print(dir(vector.features))
	vector = replace_sequence_in_record(vector, location, new_seq)

	#re-annotate the thing
	insert_loc = FeatureLocation(location.start, location.start + len(new_seq), strand=location.strand)
	destination_annotation.location = insert_loc
	destination_annotation.qualifiers['label'][0] = new_seq.name
	vector.features.append(destination_annotation)

	return vector, insert_loc



def load_inserts(inputs):
	rec_counter = 1
	inserts = []

	for this_input in inputs: 
		if os.path.isfile(this_input):
			ext = os.path.splitext(this_input)[1]
			if ext == '.fasta':
				for record in SeqIO.parse(this_input, 'fasta'):
					orig_aa_seq = record.seq
					record.seq = Seq(reverse_translate(record.seq), IUPAC.unambiguous_dna)
					assert(orig_aa_seq == record.seq.translate())
					inserts.append(record)
			elif ext == '.pdb':
				records = list(SeqIO.parse(this_input, "pdb-atom"))
				if len(records) == 1:
					record = records[0]
					name = os.path.splitext(os.path.basename(this_input))[0]
					orig_aa_seq = record.seq
					record.seq = Seq(reverse_translate(record.seq), IUPAC.unambiguous_dna)
					assert(orig_aa_seq == record.seq.translate())
					record.id=name
					record.name=name
					inserts.append(record)
				else:
					for record in records:
						name = os.path.splitext(os.path.basename(this_input))[0] + "_" + record.annotations['chain']
						orig_aa_seq = record.seq
						record.seq = Seq(reverse_translate(record.seq), IUPAC.unambiguous_dna)
						assert(orig_aa_seq == record.seq.translate())
						record.id=name
						record.name=name
						print(name)
						inserts.append(record)
			else:
				exit("extension not recognized: " + ext)
		else:
			record = SeqRecord(Seq(reverse_translate(this_input),IUPAC.unambiguous_dna), id="unknown_seq%d" % rec_counter, name="unknown_seq%d" % rec_counter, description="domesticator-optimized DNA sequence")
			rec_counter += 1
			inserts.append(record)

	return inserts

def parse_user_args():
	parser=argparse.ArgumentParser(prog='domesticator', description='The coolest codon optimizer on the block')

	parser.add_argument('--version', action='version', version='%(prog)s 0.3')

	input_parser = parser.add_argument_group(title="Input Options", description=None)
	#input_parser.add_argument("input",							 			type=str, 	default=None, 			nargs="+",	help="DNA or protein sequence(s) or file(s) to be optimized. Valid inputs are full DNA or protein sequences or fasta or genbank files. Default input is a list of protein sequences. To use a different input type, set --input_mode to the input type.")
	input_parser.add_argument("input",							 			type=str, 	default=None, 			nargs="*",	help="Protein sequence(s) or file(s) to be optimized. Valid inputs are full protein sequences and fasta and pdb files. This should be detected automatically")
	#input_parser.add_argument("--input_mode", 			dest="input_mode", 	type=str, 	default="protein_sequence", 	help="Input mode. %(default)s by default.", choices=["PDB", "DNA_fasta_file", "protein_fasta_file", "DNA_sequence", "protein_sequence"])

	cloning_parser = parser.add_argument_group(title="Cloning Options", description=None)

	cloning_parser.add_argument("--vector", "-v", 		dest="vector", 		type=str, 	default=None, 			metavar="pEXAMPLE.gb",		help="Vector used for domesticating a sequence(s) or creating new vectors")
	#cloning_parser.add_argument("--destination", "-d", 	dest="destination", type=str, 	default="INSERT", 		metavar="NAME",			help="TODO: flesh this out. Matches the dom_destination feature in the vector")

	cloning_parser.add_argument("--create_template", dest="create_template", action="store_true", default=False, help="TODO")


	optimizer_parser = parser.add_argument_group(title="Optimizer Options", description="These are only used if a vector is not specified or if create_template is set.")
	#Optimization Arguments
	optimizer_parser.add_argument("--no_opt", dest="optimize", action="store_false", default=True, help="Turn this on if you want to insert the input sequence or a naive back-translation of your protein. Not recommended (duh). Turns off all non-critical objectives and constraints")

	#optimizer options
	optimizer_parser.add_argument("--species", dest="species", default="e_coli", help="specifies the codon and dicodon bias tables to use. Defaults to %(default)s", choices=["e_coli", "s_cerevisiae", "h_sapiens"])
	optimizer_parser.add_argument("--codon_optimization_boost", dest="codon_optimization_boost", help="Give a multiplier to the codon optimizer itself. Default to %(default)f", default=1.0)
	optimizer_parser.add_argument("--harmonized", dest="harmonized", help="This will tell the algorithm to choose codons with the same frequency as they appear in nature, otherwise it will pick the best codon as often as possible.", default=False, action="store_true")

	optimizer_parser.add_argument("--avoid_hairpins", dest="avoid_hairpins", type=bool, default=True, help="Removes hairpins according to IDT's definition of a hairpin. A quicker and dirtier alternative to avoid_secondary_structure. Default to %(default)s")

	optimizer_parser.add_argument("--avoid_kmers", dest="kmers", metavar="k", default=9, type=int, help="Repeated sequences can complicate gene synthesis. This prevents repeated sequences of length k. Set to 0 to turn off. Default to %(default)d")
	optimizer_parser.add_argument("--avoid_kmers_boost", dest="avoid_kmers_boost", type=float, default=1.0, help="Give a multiplier to the avoid_kmers term. Default to %(default)f")

	optimizer_parser.add_argument("--avoid_homopolymers", dest="avoid_homopolymers", metavar="len", default=6, type=int, help="homopolymers can complicate synthesis. Prevent homopolymers longer than %(default)d by default. Specify a different length with this option. Set to 0 to turn off")

	optimizer_parser.add_argument("--avoid_patterns", dest="avoid_patterns", nargs="*", metavar="SEQUENCES", help="DNA sequence patterns to avoid", type=str)

	optimizer_parser.add_argument("--avoid_restriction_sites", dest="avoid_restriction_sites", help="Enzymes whose restriction sites you wish to avoid, such as EcoRI or BglII", nargs="*", metavar="enzy", type=str)
	optimizer_parser.add_argument("--constrain_global_GC_content", type=bool, default=True, help="TODO")
	optimizer_parser.add_argument("--global_GC_content_min", type=float, default=0.4, help="TODO")
	optimizer_parser.add_argument("--global_GC_content_max", type=float, default=0.65, help="TODO")

	optimizer_parser.add_argument("--constrain_local_GC_content", type=bool, default=True, help="TODO")
	optimizer_parser.add_argument("--local_GC_content_min", type=float, default=0.25, help="TODO")
	optimizer_parser.add_argument("--local_GC_content_max", type=float, default=0.8, help="TODO")
	optimizer_parser.add_argument("--local_GC_content_window", type=int, default=50, help="TODO")

	optimizer_parser.add_argument("--constrain_terminal_GC_content", type=bool, default=False, help="TODO")
	optimizer_parser.add_argument("--terminal_GC_content_min", type=float, default=0.5, help="TODO")
	optimizer_parser.add_argument("--terminal_GC_content_max", type=float, default=0.9, help="TODO")
	optimizer_parser.add_argument("--terminal_GC_content_window", type=int, default=16, help="TODO")

	optimizer_parser.add_argument("--constrain_CAI", type=bool, default=False, help="TODO")
	optimizer_parser.add_argument("--constrain_CAI_minimum", type=float, default=0.8, help="TODO")

	optimizer_parser.add_argument("--optimize_dicodon_frequency", type=bool, default=False, help="TODO")

	optimizer_parser.add_argument("--avoid_secondary_structure", type=bool, default=False, help="TODO")
	optimizer_parser.add_argument("--avoid_secondary_structure_max_e", type=float, default=-5.0, help="TODO")
	optimizer_parser.add_argument("--avoid_secondary_structure_boost", type=float, default=1.0, help="TODO. Has no effect if --avoid_secondary_structure is not set")

	optimizer_parser.add_argument("--avoid_initiator_secondary_structure", type=bool, default=False, help="TODO")
	optimizer_parser.add_argument("--avoid_initiator_secondary_structure_max_e", type=bool, default=-5.0, help="TODO")
	optimizer_parser.add_argument("--avoid_initiator_secondary_structure_boost", type=float, default=5.0, help="TODO. Has no effect if --avoid_5'_secondary_structure is not set")


	ordering_parser = parser.add_argument_group(title="Ordering Options", description=None)
	ordering_parser.add_argument("--order_type", choices=["gBlocks","genes"], default=None, help="Choose how you'll order your sequences through IDT and you'll get a file called to_order.fasta that you can directly submit")


	output_parser = parser.add_argument_group(title="Output Options", description=None)
	#Output Arguments
	output_parser.add_argument("--output_mode", dest="output_mode", default="terminal", choices=['terminal', 'fasta', 'genbank', 'none'], help="Default: %(default)s\n Choose a mode to export complete sequences in the vector, if specified.")
	output_parser.add_argument("--output_filename", dest="output_filename", help="defaults to %(default)s.fasta or %(default)s.gb", default="domesticator_output")

	return parser.parse_args()



if __name__ == "__main__":

	args = parse_user_args()

	destination = "DOMESTICATOR_INSERT"

	if args.create_template:

		

		placeholder = SeqRecord(Seq("cgctatgcgaacaaaattgaactggaacgc", alphabet=IUPAC.unambiguous_dna), name=destination)


		if args.vector:
			base, ext = os.path.splitext(os.path.basename(args.vector))

			if destination in base:
				output_filename = base + ext
			else:
				output_filename = base + "_" + destination + ext
			naive_construct, objectives, constraints, insert_location = load_template(args.vector, placeholder, destination)
		else:
			output_filename = destination + ".gb"
			objectives = []
			constraints = []
			naive_construct = placeholder
			whole_seq_feat = SeqFeature()
			whole_seq_feat.type = "misc_feature"
			whole_seq_feat.qualifiers['label'] = [destination]
			whole_seq_feat.location = FeatureLocation(0,len(placeholder),strand=1)
			naive_construct.features.append(whole_seq_feat)

		dest_feat = find_annotation(naive_construct, placeholder.name)
		dest_loc = Location.from_biopython_location(dest_feat.location)


		user_objectives, user_constraints = load_user_options(args, dest_loc)
	

		objectives += user_objectives
		constraints += user_constraints

		problem = DnaOptimizationProblem(str(naive_construct.seq), constraints=constraints, objectives=objectives)

		domesticator_record = problem.to_record()

		mature_construct = naive_construct
		mature_construct.features.extend(domesticator_record.features)

		SeqIO.write([mature_construct], output_filename, "genbank")
		#BAD BAD
		exit("exported " + output_filename)




	inserts = load_inserts(args.input)

	#now load all the custom global constraints and objectives?

	outputs = []

	for insert in inserts:
		if args.vector:

			#args.output_mode = 'none'
			naive_construct, objectives, constraints, insert_location= load_template(args.vector, insert, destination)
		else:
			#wasn't given a vector
			naive_construct = insert
			objectives = []
			constraints = []

			#insert_location = Location(0, len(insert))
			insert_location = FeatureLocation(0, len(insert))

			objectives, constraints = load_user_options(args, insert_location)

		before_aa = insert_location.extract(naive_construct.seq).translate()
		problem = DnaOptimizationProblem(str(naive_construct.seq), constraints=constraints, objectives=objectives)

		if args.optimize:
			##optimize
			try_num = 0
			max_tries = 1
			while True:
				try:
					print("attempting optimization of " + naive_construct.name)
					problem.resolve_constraints()
					problem.optimize()
					problem.resolve_constraints(final_check=True)
					break
				except NoSolutionError:
					try_num += 1
					if try_num < max_tries:
						problem.max_random_iters += 1000
						print(problem.constraints_text_summary())
						print(problem.objectives_text_summary())
						print("optimization of %s failed! Attempt %d of %d. Trying again with %d random iters" % (naive_construct.name, try_num + 1, max_tries, problem.max_random_iters))
					else:
						print("optimization of %s failed!" % (naive_construct.name))
						#problem.sequence = ""
						break


		print(problem.constraints_text_summary())
		print(problem.objectives_text_summary())

		mature_construct = naive_construct
		mature_construct.seq = Seq(problem.sequence, alphabet=IUPAC.unambiguous_dna)
		mature_construct.name = insert.name
		mature_construct.id = insert.name

		after_aa = insert_location.extract(mature_construct.seq).translate()

		#print(before_aa, after_aa)
		assert(before_aa == after_aa)

		if args.vector:
			template_basename = os.path.basename(args.vector)

			if destination in template_basename:
				custom_filename = template_basename.replace(destination, insert.name)
			else:
				base, ext = os.path.splitext(template_basename)
				custom_filename = base + "_" + insert.name + ext

			mature_construct.name = os.path.splitext(custom_filename)[0]
			mature_construct.id = mature_construct.name
			SeqIO.write([mature_construct], custom_filename, "genbank")

		outputs.append(mature_construct)

		#take vector name and replace the destination name with the insert name?

		#does this work right?

	#SO ordering... How does ordering work. 

	#REMEMBER to set the description to "" for easy ordering
	if args.order_type == "gBlocks":
		with open("order.fasta", "a+") as order:
			SeqIO.write([SeqRecord(find_annotation(output, "gBlock_to_order").location.extract(output.seq),id=output.id,name=output.name,description="") for output in outputs], order, "fasta")
	elif args.order_type == "genes":
		#simply output the thing having been inserted. 
		with open("order.fasta", "a+") as order:
			SeqIO.write([SeqRecord(find_annotation(output, "gene_to_order").location.extract(output.seq),id=output.id,name=output.name,description="") for output in outputs], order, "fasta")


	#time to handle IO
	if args.output_mode == 'none':
		pass
	elif args.output_mode == 'terminal':
		for output in outputs:
			output.description = ""
			print(output.format("fasta"))
	elif args.output_mode:
		for output in outputs:
			output.description = ""
		SeqIO.write(outputs, args.output_filename, args.output_mode)
