from chemdataextractor.model.units import Unit, Dimension, QuantityModel, Mass, Time, AmountOfSubstance, PowerModel, LengthModel, TimeModel, Minute, Hour, Gram
from chemdataextractor.model.contextual_range import SectionRange, SentenceRange
from chemdataextractor.model import BaseModel, Compound, ModelType, StringType, ModelList, ListType, InferredProperty
from chemdataextractor.parse.elements import R, I, W, Optional, Not, Every, Group, SkipTo
from chemdataextractor.parse.actions import join, flatten
from chemdataextractor.parse.auto import AutoSentenceParser
from chemdataextractor.parse.quantity import magnitudes_dict, value_element, extract_value
from chemdataextractor.parse.cem_factory import _CemFactory
from chemdataextractor.parse.auto_dependency import AutoDependencyParser
from lxml.etree import strip_tags
import copy


# Helper functions
def restrict_value_range(value_range):
    # value_range is going to be a tuple
    def _internal(result):
        try:
            for el in result:
                if el.tag == "raw_value":
                    raw_value = el.text
                    value = extract_value(raw_value)
                    if value is not None and len(value):
                        if len(value) == 1:
                            value = [value[0], value[0]]
                        if value[0] > value_range[0] and value[1] < value_range[1]:
                            return True
        except IndexError:
            pass
        return False
    return _internal


def restricted_value_element(value_range):
    def new_value_element(*args, **kwargs):
        return value_element(*args, **kwargs).with_condition(restrict_value_range(value_range))
    return new_value_element


def unwrap_tag(tag):
    def unwrap(tokens, start, result):
        """Join tokens into a single string with spaces between."""
        for e in result:
            strip_tags(e, tag)
        return result
    return unwrap


def value_units_inferrer(string, instance):
    raw_value = instance.raw_value.replace(" ", "").replace("(", "").replace(")", "") if instance.raw_value is not None else ""
    raw_units = instance.raw_units.replace(" ", "").replace("(", "").replace(")", "") if instance.raw_units is not None else ""
    return " ".join([raw_value, raw_units])


# Set up units
class Fraction(Dimension):
    pass


class FractionModel(QuantityModel):
    dimensions = Fraction()


class FractionalUnit(Unit):
    def __init__(self, magnitude=0.0, powers=None):
        super(FractionalUnit, self).__init__(Fraction(), magnitude, powers)


class Percent(FractionalUnit):
    def convert_value_to_standard(self, value):
        return value

    def convert_value_from_standard(self, value):
        return value

    def convert_error_to_standard(self, error):
        return error

    def convert_error_from_standard(self, error):
        return error


class PartsPerMillion(FractionalUnit):
    def convert_value_to_standard(self, value):
        return value / 10000.

    def convert_value_from_standard(self, value):
        return value * 10000.

    def convert_error_to_standard(self, error):
        return error / 10000.

    def convert_error_from_standard(self, error):
        return error * 10000.


units_dict = {R(r'(\%)|([Pp]ercent)', group=0): Percent}
Fraction.units_dict = units_dict
Fraction.standard_units = Percent()

Mass.units_dict[R('gcat', group=0)] = Gram

magnitudes_dict[R('cat', group=0)] = 0.
magnitudes_dict[R('H2', group=0)] = 0.

# No valid cases where it's not minutes or hours, so remove others
keys_to_remove = []
for key, value in Time.units_dict.items():
    if value != Minute and value != Hour:
        keys_to_remove.append(key)

for key in keys_to_remove:
    Time.units_dict.pop(key)

# Set up parsing for compounds
joining_characters = R(r'^\@|\/|:|[-–‐‑‒–—―]$')
cem_factory = _CemFactory(joining_characters=joining_characters)
cem = cem_factory.cem
other_solvent = cem_factory.other_solvent

chem_name = Every([(cem), Not(I('Hydrogen')), Not(W('H2')), Not(W('H+')), Not(W('STH')),
                  Not(I('Methanol')), Not(I('Glycerol')), Not(I('Ethanol')), Not(I('Oxygen')), Not(I('Water')), Not(W('O2')), Not(I('Acetaldehyde')),
                  Not(I('Xe')), Not(W('H2O2'))])

good_additive_names = copy.copy(other_solvent) ^ W('HCl') ^ I('ethanol') ^ I('methanol') ^ (I('oxalic') + I('acid'))
additive_names = Group(Group(Every([good_additive_names, Not(I('water')), Not(I('H2O'))])).add_action(join)('names'))('compound')
skip_section_phrase = (R(r'^[Cc]haracter') ^ R(r'^[Ss]pectr') ^ R(r'^[Mm]icroscop')
                       ^ I('Photoabsorbence') ^ R(r'^[Mm]orphol') ^ I('Optical') ^ R(r'^[Mm]echanis') ^ R(r'^[Cc]haracteri')
                       ^ (I('Charge') + Optional(W('-')) + I('Transfer')) ^ I('Preparation')
                       ^ (I('Oxidation') + I('State'))
                       ^ (I('Carbon') + I('Template'))
                       ^ ((I('Dynamic') ^ I('Kinetic')) + I('Model'))
                       ^ (I('Structural') + I('Analysis'))
                       ^ I('Preparation')
                       ^ I('Synthesis')
                       ^ W('XPS') ^ W('FTIR') ^ W('TEM')
                       ^ W('SEM') ^ W('EDX') ^ W('EPR')
                       ^ R('[Ee]lectrochemical')
                       ^ I('Spectroscopy')
                       ^ I('Construction')
                       ^ I('Deposition')
                       ^ I('Reagents')
                       ^ R('[Dd]egredation')
                       ^ (I('Methylene') + I('Blue'))
                       ^ R('[Tt]oxic')
                       ^ I('Photolysis')
                       ^ W('MB')
                       ^ R('[Rr]eduction')
                       ^ W('CO2')
                       ^ I('photocurrent'))
allow_section_phrase = R(r'[Aa]ctivit') ^ (I("Hydrogen") + I("Generation")) ^ (R(r"^[Pp]hotocatal") + I("Testing")) ^ (R(r"^[Pp]hotocatal") + R(r"^[Pp]ropert"))


class SimpleCompoundParser(AutoSentenceParser):
    root = Group(chem_name)


Compound.parsers = [SimpleCompoundParser()]

# Set up Additive parsing
additive_skip_phrase = I('Electrolyte') ^ I('XRD') ^ R('^[Ww]ash')


class Additive(BaseModel):
    compound = ModelType(Compound, contextual=False, required=True, never_merge=True)
    specifier = StringType(parse_expression=I('aqueous') ^ I('solution') ^ I('water') ^ I('sacrificial'), required=False, contextual=False, never_merge=True)
    parsers = [AutoDependencyParser(chem_name=additive_names, skip_phrase=additive_skip_phrase, skip_section_phrase=skip_section_phrase, allow_section_phrase=allow_section_phrase)]


# Set up co-catalyst parsing
cocatalyst_phrase = (I("cocatalyst") | (I("co") + I("-") + I("catalyst")) | (I("Photocatalyst") + SkipTo(I("loaded")) + I("loaded") + I("with"))).add_action(join)
cocatalyst_skip_phrase = (((I("Without") ^ I("No")) + cocatalyst_phrase) ^ I("absence") ^ (cocatalyst_phrase + Optional(W("-")) + I("free"))).add_action(join)


class Cocatalyst(BaseModel):
    compound = ModelType(Compound, contextual=False, required=True, never_merge=True)
    specifier = StringType(parse_expression=cocatalyst_phrase, required=False, contextual=False, ignore_when_merging=True)
    parsers = [AutoDependencyParser(chem_name=chem_name, skip_phrase=cocatalyst_skip_phrase, skip_section_phrase=skip_section_phrase, allow_section_phrase=allow_section_phrase)]


# Set up photocatalyst parsing
photocatalyst_phrase = Group(I("photocatalyst") | (I("photo") + I("-") + I("catalyst"))).add_action(flatten)("specifier")
photocatalyst_name = Group(
        (photocatalyst_phrase + SkipTo(Group(chem_name)).hide() + Group(chem_name))
        ^ (Group(chem_name) + Optional(SkipTo(photocatalyst_phrase).hide() + photocatalyst_phrase))).add_action(unwrap_tag("compound"))("compound")


class SimplePhotocatalystParser(AutoSentenceParser):
    # This may look suboptimal as a way to do it, but if you combine these two or clauses into one big thing with
    # two optionals, with a skip to inbetween, the first optional thing is never encountered as the parse expression will
    # always skip to the chem_name
    root = Group(
        (photocatalyst_phrase + SkipTo(Group(chem_name)).hide() + Group(chem_name))
        ^ (Group(chem_name) + Optional(SkipTo(photocatalyst_phrase).hide() + photocatalyst_phrase))).add_action(unwrap_tag("compound"))("root_phrase")


# We can't do the same thing as with cocatalyst because we do some special handling for compounds...
class Photocatalyst(Compound):
    # specifier = StringType(parse_expression=photocatalyst_phrase, required=True, requiredness=0.3, contextual=False, never_merge=True)
    specifier = StringType(parse_expression=photocatalyst_phrase, required=False, requiredness=0.0, contextual=False, never_merge=True)
    parsers = [SimpleCompoundParser()]


# Set up models for photocatalytic activities
wavelength_skip_phrase = R("^[Ss]pectr") ^ I("Beam") ^ (I("Electrochemical") + I("Impedance")) ^ W("EIS")


hydrogen_evolution_parse_expression = (((W("H2") | I("Hydrogen"))
                                        + Optional(I("-"))
                                        + (I("Evolution")
                                           | I("Generation")
                                           | I("Production")
                                           | I("Yield")
                                           | I("Photoactivity")
                                           | (I("gas") + I("at") + I("a") + I("rate"))
                                           | I("gas")
                                           | (SkipTo(I("Rate")) + I("Rate"))))
                                       | (W("HER") + I("rate"))).add_action(join)

skip_irradiation_time_phrase = (I("Before") + SkipTo(R("radiat")) + R("radiat")) ^ I("degas") ^ R("^stabili") ^ R("sonic")

name_expression = ((I("black") + ((I("light") + I("tubes")) ^ I("lamp")))
                   ^ ((W("Xe") ^ W("Hg") ^ W("Ar") ^ I("xenon") ^ I("mercury") ^ I("argon") ^ I("Tungsten") ^ (I("simulated") + I("sunlight"))) + Optional(I("Visible") ^ I("ultraviolet") ^ I("arc")) + (R("lamps?") ^ R("lights?")))
                   ^ ((I("near") + Optional(W("-")) + W("UV") + Optional(W("/")))
                      + ((I("monochromatic") + I("excitation")) ^ I("visible") ^ W("UV")) + Optional(W("-")) + (R("lamps?") ^ R("lights?")))
                   ^ (W('UV') + Optional(W('-')) + I('Visible') + (R("lamps?") ^ R("lights?")))
                   ^ (I("monochromatic") + I("excitation") + (R("lamps?") ^ R("lights?")))
                   ^ W("LED")
                   ).add_action(join)


class LightSourceWavelength(LengthModel):
    raw_value = StringType(required=True, contextual=False)
    raw_units = StringType(required=True, contextual=False)
    specifier = StringType(parse_expression=I("wavelength") ^ R("λ") ^ I("excitation") ^ (hydrogen_evolution_parse_expression + SkipTo(W("at"))),
                           required=True, ignore_when_merging=True, contextual=False)
    val_units = InferredProperty(StringType(), origin_field="raw_value", inferrer=value_units_inferrer)
    parsers = [AutoDependencyParser(primary_keypath="specifier", skip_phrase=wavelength_skip_phrase,
                                    skip_section_phrase=skip_section_phrase, allow_section_phrase=allow_section_phrase,
                                    value_phrase_constructor=restricted_value_element((200, 900)))]


class LightSourcePower(PowerModel):
    raw_value = StringType(required=True, contextual=False)
    raw_units = StringType(required=True, contextual=False)
    specifier = StringType(parse_expression=I("lamp") ^ I("intensity") ^ R('[Rr]adiat') ^ I("illumination") ^ name_expression,
                           required=True, ignore_when_merging=True, contextual=False)
    val_units = InferredProperty(StringType(), origin_field="raw_value", inferrer=value_units_inferrer)
    parsers = [AutoDependencyParser(primary_keypath="specifier", skip_section_phrase=skip_section_phrase, allow_section_phrase=allow_section_phrase)]


class IrradiationTime(TimeModel):
    raw_value = StringType(required=True, contextual=False)
    raw_units = StringType(required=True, contextual=False)
    specifier = StringType(parse_expression=R("radiat") ^ I("during") ^ R("^illuminat") ^ R("light") ^ hydrogen_evolution_parse_expression,
                           required=True, ignore_when_merging=True, contextual=False)
    val_units = InferredProperty(StringType(), origin_field="raw_value", inferrer=value_units_inferrer)
    parsers = [AutoDependencyParser(primary_keypath="specifier", skip_phrase=skip_irradiation_time_phrase,
                                    skip_section_phrase=skip_section_phrase, allow_section_phrase=allow_section_phrase)]


filter_expression = (((Optional(W("R64") ^ I("Optical")) + I("Cutoff"))
                     ^ (W("BP") + W("-") + W("42") + W("HOYA"))
                     ^ (I("optical") + I("band") + I("pass"))
                     ^ I("UV") + Optional(Optional(W("-")) + I("Cutoff"))) + I("filter")).add_action(join)

lightsource_skip_phrase = R("^[Ss]pectroscop") ^ I("Beam") ^ (I("Electrochemical") + I("Impedance")) ^ W("EIS")


class LightSource(BaseModel):
    name = StringType(parse_expression=name_expression, required=True, contextual=False)
    filters = StringType(parse_expression=filter_expression, required=False, contextual=False)
    parsers = [AutoDependencyParser(primary_keypath="name", skip_phrase=lightsource_skip_phrase, skip_section_phrase=skip_section_phrase, allow_section_phrase=allow_section_phrase)]


merging_range = 2 * SectionRange()
cocatalyst_merging_range = 2 * SentenceRange()

activity_skip_phrase = skip_section_phrase

is_reported_value_phrase = ((I("reported") + I("by")) ^ (I("et") + I("al"))).add_action(join)


class _PhotocatalyticActivity(BaseModel):
    cocatalyst = ModelType(Cocatalyst, contextual=True, contextual_range=merging_range)
    # compound = ModelType(Compound, contextual=True, required=True, contextual_range=merging_range)
    compound = ModelType(Photocatalyst, contextual=True, required=True, contextual_range=merging_range)
    additives = ListType(ModelType(Additive), contextual=True, contextual_range=merging_range)
    light_source = ModelType(LightSource, contextual=True, contextual_range=merging_range)
    light_source_wavelength = ModelType(LightSourceWavelength, contextual=True, contextual_range=merging_range)
    light_source_power = ModelType(LightSourcePower, contextual=True, contextual_range=merging_range)
    irradiation_time = ModelType(IrradiationTime, contextual=True, contextual_range=merging_range)
    no_cocatalyst = StringType(parse_expression=cocatalyst_skip_phrase, never_merge=True, contextual=False)
    raw_value = StringType(required=True, contextual=False)
    raw_units = StringType(required=True, contextual=False)
    val_units = InferredProperty(StringType(), origin_field="raw_value", inferrer=value_units_inferrer)
    is_reported_value = StringType(parse_expression=is_reported_value_phrase, contextual=False, never_merge=True, ignore_when_merging=True)

    def contextual_range(self, field_name):
        if self.is_reported_value is not None:
            return 0. * SentenceRange()
        else:
            return self.fields[field_name].contextual_range


class ApparentQuantumYield(FractionModel, _PhotocatalyticActivity):
    specifier = StringType(parse_expression=(W("AQY") | (I("Apparent") + I("Quantum") + I("Yield")).add_action(join)),
                           required=True, ignore_when_merging=True)
    parsers = [AutoDependencyParser(chem_name=chem_name, skip_section_phrase=skip_section_phrase, allow_section_phrase=allow_section_phrase, skip_phrase=activity_skip_phrase)]


class SolarToHydrogen(FractionModel, _PhotocatalyticActivity):
    specifier = StringType(parse_expression=(W("STH") | (I("Solar") + Optional(I("-")) + I("to") + Optional(I("-")) + I("Hydrogen")).add_action(join)),
                           required=True, ignore_when_merging=True)
    parsers = [AutoDependencyParser(chem_name=chem_name, skip_section_phrase=skip_section_phrase, allow_section_phrase=allow_section_phrase, skip_phrase=activity_skip_phrase)]


class HydrogenEvolution(QuantityModel, _PhotocatalyticActivity):
    dimensions = AmountOfSubstance() / (Mass() * Time())
    specifier = StringType(parse_expression=hydrogen_evolution_parse_expression,
                           required=True, ignore_when_merging=True)
    parsers = [AutoDependencyParser(chem_name=chem_name, skip_section_phrase=skip_section_phrase, allow_section_phrase=allow_section_phrase, skip_phrase=activity_skip_phrase)]


class HydrogenEvolution2(QuantityModel, _PhotocatalyticActivity):
    dimensions = AmountOfSubstance() / Mass()
    specifier = StringType(parse_expression=hydrogen_evolution_parse_expression,
                           required=True, ignore_when_merging=True)
    parsers = [AutoDependencyParser(chem_name=chem_name, skip_section_phrase=skip_section_phrase, allow_section_phrase=allow_section_phrase, skip_phrase=activity_skip_phrase)]


class HydrogenEvolution3(QuantityModel, _PhotocatalyticActivity):
    dimensions = AmountOfSubstance() / Time()
    specifier = StringType(parse_expression=hydrogen_evolution_parse_expression,
                           required=True, ignore_when_merging=True)
    parsers = [AutoDependencyParser(chem_name=chem_name, skip_section_phrase=skip_section_phrase, allow_section_phrase=allow_section_phrase, skip_phrase=activity_skip_phrase)]


# Cleaning results
def remove_lower_confidence(results):
    # Should we do anything wrt boosting recall by having this bias towards
    # those records with more subrecords? e.g. score as num records * confidence
    filtered_results = ModelList()
    results_by_type = {}

    for result in results:
        if type(result) in results_by_type.keys():
            results_by_type[type(result)].append(result)
        else:
            results_by_type[type(result)] = [result]

    for _, type_results in results_by_type.items():
        results_by_value = {}

        for result in type_results:
            if hasattr(result, "value"):
                str_value = str(result.value)
                if str_value in results_by_value.keys():
                    results_by_value[str_value].append(result)
                else:
                    results_by_value[str_value] = [result]
            else:
                filtered_results.append(result)

        for _, value_results in results_by_value.items():
            print([result.serialize() for result in value_results])
            # As a tie-breaker, we'd rather take results that are from in the middle,
            # because it's harder to get high confidences in the middle of a document.

            value_results.reverse()
            max_result = max(value_results, key=lambda x: x.total_confidence())
            filtered_results.append(max_result)

    return filtered_results


# If we found the phrase "no co-catalyst" or "pure", we want to ensure that
# we remove any co-catalysts that we associated with the record
def remove_cocatalyst_if_nococatalyst(results):
    filtered_results = ModelList()
    for result in results:
        if isinstance(result, _PhotocatalyticActivity):
            if result.no_cocatalyst:
                result.cocatalyst = None
        filtered_results.append(result)
    return filtered_results


# To work with CDE's framework, we have three different "hydrogen evolution"s with different unit types
# With this function, we remove any that are "subsets", i.e. moles per hour if we found something with moles per hour per gram in the same document
def remove_hydrogenevolution(results):
    hydrogen_evolution_results = ModelList()
    hydrogen_evolution_subset_results = ModelList()
    other_results = ModelList()
    for result in results:
        if isinstance(result, HydrogenEvolution):
            hydrogen_evolution_results.append(result)
        elif isinstance(result, HydrogenEvolution2) or isinstance(result, HydrogenEvolution3):
            hydrogen_evolution_subset_results.append(result)
        else:
            other_results.append(result)
    filtered_results = other_results
    filtered_results.extend(hydrogen_evolution_results)
    for result in hydrogen_evolution_subset_results:
        should_remove = False
        if hasattr(result, "raw_value"):
            for parent_result in hydrogen_evolution_results:
                if (hasattr(parent_result, "raw_value")
                   and result.raw_value == parent_result.raw_value
                   and result.raw_units.replace(" ", "") in parent_result.raw_units.replace(" ", "")):
                    should_remove = True
                    break
        if not should_remove:
            filtered_results.append(result)
    return filtered_results


def filter_results(results):
    return remove_lower_confidence(remove_hydrogenevolution(remove_cocatalyst_if_nococatalyst(results)))


# Filtering titles
title_good_phrase = I("Water") + Not(I("Treatment"))
title_bad_phrase = R("[Dd]egradation") ^ I("decomposition") ^ I("Photoconversion") ^ (I("Water") + I("Treatment")) ^ R('[Cc]athode') ^ R('[Aa]node') ^ W("CO2")


# Things on the left tend to be in the names of result-y sections and the things on the right tend to be in the names of method-y sections
adjascent_sections = [(["photocatal", "experiment", "result", "hydrogen production", "h2 production", "h2 generation", "h2 evolution", "hydrogen evolution", "conclusion"], ["photocatal", "experiment", "hydrogen production", "h2 production", "h2 evolution", "h2 generation", "hydrogen evolution"])]


def is_valid_document(doc):
    titles = doc.titles
    for title in titles:
        for title_sent in title.sentences:
            for _ in title_good_phrase.scan(title_sent.tokens):
                return True
            for _ in title_bad_phrase.scan(title_sent.tokens):
                return False
    return True
