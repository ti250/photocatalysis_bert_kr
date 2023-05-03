import os

from photocatalyst_models import (
    Additive,
    Cocatalyst,
    Photocatalyst,
    LightSource,
    LightSourceWavelength,
    LightSourcePower,
    IrradiationTime,
    cocatalyst_skip_phrase,
    is_reported_value_phrase,
    skip_section_phrase,
    allow_section_phrase,
    activity_skip_phrase,
    ApparentQuantumYield,
    HydrogenEvolution
)

from chemdataextractor.parse.multi_turn_qa import MultiTurnQAParser, Q
from chemdataextractor.parse.auto_dependency import AutoDependencyParser
from chemdataextractor.model import ModelType, StringType, BaseModel, ListType, InferredProperty, FloatType
from chemdataextractor.parse import Optional, I, W, join, SkipTo, Group
from chemdataextractor.parse.elements import NoMatch
from chemdataextractor.model.contextual_range import SectionRange, SentenceRange
from chemdataextractor.doc.text import Sentence
from chemdataextractor.parse.quantity import infer_value
from chemdataextractor.parse.quantity import value_element
from chemdataextractor.parse.auto import construct_unit_element
from pprint import pprint

model_name = "deepset/deberta-v3-large-squad2"
model = None

if os.getenv("BERT_MODEL_NAME") is not None:
    model_name = os.getenv("BERT_MODEL_NAME")

handle_impossible_answer = True
photocat_context_range = 10
photocat_include_heading = True

cocatalyst_context_range = 7
cocatalyst_nomerge_range = 7

additive_context_range = 1
additive_nomerge_range = 1

combined_dimensions = HydrogenEvolution.dimensions

merging_range = 2 * SectionRange()
cocatalyst_merging_range = 2 * SentenceRange()

aqy_specifier = (W("AQY") | (I("Apparent") + I("Quantum") + I("Yield")).add_action(join))
sth_specifier = (W("STH") | (I("Solar") + Optional(I("-")) + I("to") + Optional(I("-")) + I("Hydrogen")).add_action(join))
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


def value_units_inferrer(string, instance):
    if instance.raw_value:
        raw_value = instance.raw_value.replace(" ", "").replace("(", "").replace(")", "")
        raw_units = instance.raw_units.replace(" ", "").replace("(", "").replace(")", "")
        return " ".join([raw_value, raw_units])


def infer_raw_value(string, instance):
    if string:
        raw_value = string.split(" ", 1)[0]
        return raw_value


def infer_raw_units(string, instance):
    if string:
        raw_units = string.split(" ", 1)[-1]
        return raw_units


def clean_additive_action(answer):
    if "water" in answer:
        answer = answer.replace("water", "")
        answer = answer.strip("/ ")
    return answer


def is_not_banned_cem(answer):
    print(answer)
    banned_chems = [
        "h2", "o2", "h+", "sth", "methanol", "glycerol", "ethanol", "oxygen",
        "water", "acetaldehyde", "xe", "h2o2", "absence", "hydrogen"
    ]
    print(answer.lower() in banned_chems)
    if answer.lower() in banned_chems:
        return False
    return True


class AutoDependencyParserWithLaxValuePhrase(AutoDependencyParser):
    @property
    def _value_phrase(self):
        value_phrase_constructor = self.value_phrase_constructor
        if self.value_phrase_constructor is None:
            value_phrase_constructor = value_element
        if hasattr(self.model, "dimensions") and not self.model.dimensions:
            return value_phrase_constructor()
        unit_element = Group(construct_unit_element(self.model.dimensions)('raw_units'))
        return value_phrase_constructor(unit_element)


class PhotocatalyticActivity(BaseModel):
    dimensions = combined_dimensions

    cocatalyst = ModelType(
        Cocatalyst,
        contextual=True,
        contextual_range=merging_range,
        parse_expression=Q(
            "What co-catalytic material was used alongside {} when the {} was found to be {}?",
            ["compound.names", "specifier", "val_units"],
            num_preceding_sentences=cocatalyst_context_range,
            include_heading=False,
            confidence_threshold=0.5,
            no_merge_range=cocatalyst_nomerge_range,
        ).with_condition(is_not_banned_cem).add_action(clean_additive_action)(".compound.names"),
    )
    compound = ModelType(
        Photocatalyst,
        parse_expression=Q(
            "What photocatalytic material has a {} of {}?",
            ["specifier", "val_units"],
            num_preceding_sentences=photocat_context_range,
            include_heading=photocat_include_heading,
        ).with_condition(is_not_banned_cem)(".names"),
        contextual=True,
        required=True,
        contextual_range=merging_range
    )
    additives = ListType(
        ModelType(Additive),
        contextual=True,
        contextual_range=merging_range,
        parse_expression=Q(
            "What chemical was dissolved in the solution when measuring the {} of {} to be {}?",
            ["specifier", "compound.names", "val_units"],
            num_preceding_sentences=additive_context_range,
            include_heading=False,
            confidence_threshold=0.7,
            no_merge_range=additive_nomerge_range,
        ).with_condition(is_not_banned_cem).add_action(clean_additive_action)(".compound.names"),
    )
    light_source = ModelType(LightSource, contextual=True, contextual_range=merging_range)
    light_source_wavelength = ModelType(LightSourceWavelength, contextual=True, contextual_range=merging_range)
    light_source_power = ModelType(LightSourcePower, contextual=True, contextual_range=merging_range)
    irradiation_time = ModelType(IrradiationTime, contextual=True, contextual_range=merging_range)
    no_cocatalyst = StringType(parse_expression=cocatalyst_skip_phrase, never_merge=True, contextual=False)
    raw_value = StringType(required=True, contextual=False)
    raw_units = StringType(required=True, contextual=False)
    value = InferredProperty(
        ListType(FloatType(), sorted_=True),
        origin_field='raw_value',
        inferrer=infer_value,
        contextual=True
    )
    val_units = InferredProperty(StringType(), origin_field="raw_value", inferrer=value_units_inferrer)
    specifier = StringType(parse_expression=Group(hydrogen_evolution_parse_expression).add_action(join),
                           required=True, ignore_when_merging=True)
    is_reported_value = StringType(parse_expression=is_reported_value_phrase, contextual=False, never_merge=True, ignore_when_merging=True)

    parsers = [MultiTurnQAParser(
        skip_section_phrase=skip_section_phrase,
        allow_section_phrase=allow_section_phrase,
        skip_phrase=activity_skip_phrase,
        confidence_threshold=0.0,
        model_name=model_name,
        model=model,
        batch_size=16,
        enable_single_sentence=False,
        manual_parser=AutoDependencyParserWithLaxValuePhrase(chem_name=NoMatch()),
        handle_impossible_answer=handle_impossible_answer,
    )]

    def contextual_range(self, field_name):
        if self.is_reported_value is not None:
            return 0. * SentenceRange()
        else:
            return self.fields[field_name].contextual_range


class PhotocatalyticEfficiency(PhotocatalyticActivity):
    dimensions = ApparentQuantumYield.dimensions

    specifier = StringType(parse_expression=Group(aqy_specifier ^ sth_specifier).add_action(join),
                           required=True, ignore_when_merging=True)
    parsers = [MultiTurnQAParser(
        skip_section_phrase=skip_section_phrase,
        allow_section_phrase=allow_section_phrase,
        skip_phrase=activity_skip_phrase,
        confidence_threshold=0.0,
        model_name=model_name,
        model=model,
        batch_size=16,
        enable_single_sentence=False,
        manual_parser=AutoDependencyParserWithLaxValuePhrase(chem_name=NoMatch()),
        handle_impossible_answer=handle_impossible_answer,
    )]


if __name__ == "__main__":
    while True:
        text = input("Please enter a sentence: ")
        sent = Sentence(text)
        sent.models = [PhotocatalyticActivity, PhotocatalyticEfficiency]
        PhotocatalyticEfficiency.parsers[0].enable_single_sentence = True
        PhotocatalyticActivity.parsers[0].enable_single_sentence = True
        for record in sent.records:
            pprint(record.serialize())
            print(f"Confidence: {record.total_confidence()}")
