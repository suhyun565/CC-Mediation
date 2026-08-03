"""
utils.py
========

CONFLICT_TABLE for the Cultural Conflict mediation evaluation pipeline.

The table is restricted to the three ETHNOCENTRIC stages of Bennett's
Developmental Model of Intercultural Sensitivity (DMIS): Denial, Defense,
and Minimization. Each entry contains:

  description : One-sentence characterization of what the conflict looks
                like at this stage.
  mediation   : One-sentence statement of what a third-party mediator
                should do at this stage.

Sources
-------

  Bennett, M. J. (1986). A developmental approach to training for
    intercultural sensitivity. International Journal of Intercultural
    Relations, 10(2), 179-195.
  Bennett, M. J. (1993). Towards ethnorelativism: A developmental model of
    intercultural sensitivity. In R. M. Paige (Ed.), Education for the
    intercultural experience (pp. 21-71). Intercultural Press.
  Bennett, M. J. (2004). From ethnocentrism to ethnorelativism. In J. S.
    Wurzel (Ed.), Toward multiculturalism: A reader in multicultural
    education. Intercultural Resource Corporation.
  Bennett, J., & Bennett, M. (2004). Developing intercultural sensitivity:
    An integrative approach to global and domestic diversity. In D. Landis,
    J. Bennett, & M. Bennett (Eds.), Handbook of intercultural training
    (3rd ed., pp. 147-165). Sage.
  Bennett, M. J. (2011). A Developmental Model of Intercultural Sensitivity
    [12pp document with quotes, rev. 2011]. Intercultural Development
    Research Institute.
    https://www.idrinstitute.org/wp-content/uploads/2018/02/FILE_Documento_Bennett_DMIS_12pp_quotes_rev_2011.pdf
"""

CONFLICT_TABLE = {
    "Denial": {
        "description": (
            "Speakers fail to register cultural difference as a "
            "meaningful category and treat the disagreement as if no "
            "cultural dimension were in play."
        ),
        "mediation": (
            "Surface the existence of a cultural dimension the speakers "
            "have not yet noticed using low-stakes, curiosity-arousing "
            "framing, while providing high support and avoiding deep "
            "value contrasts that would push the speakers into Defense."
        ),
    },
    "Defense": {
        "description": (
            "Speakers experience cultural difference in a polarized "
            "us-versus-them frame and rely on simplistic, evaluative "
            "stereotypes of the other party."
        ),
        "mediation": (
            "Depolarize the conversation by avoiding direct cultural "
            "contrasts, emphasizing shared concerns and goals, and "
            "framing the in-group/out-group dynamic as a universal "
            "human pattern rather than a personal failing of either "
            "party."
        ),
    },
    "Minimization": {
        "description": (
            "Speakers subsume cultural difference under an assumed "
            "universal sameness and treat their own cultural patterns "
            "as the neutral baseline that all parties supposedly share."
        ),
        "mediation": (
            "Turn each speaker's attention onto their OWN cultural "
            "patterns, making visible that what they treat as universal "
            "common sense is in fact specific to their group, and "
            "surface the privilege of the dominant party in the "
            "conversation."
        ),
    },
}