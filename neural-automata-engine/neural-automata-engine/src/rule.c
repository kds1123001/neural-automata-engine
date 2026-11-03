#include <string.h>

#include "nae/rules.h"

typedef const nae_rule *(*rule_getter)(void);

static const rule_getter registry[] = {
    nae_rule_life,
    nae_rule_neural,
};

const nae_rule *nae_rule_find(const char *name)
{
    size_t i;

    if (!name)
        return NULL;
    for (i = 0; i < sizeof registry / sizeof registry[0]; ++i) {
        const nae_rule *rule = registry[i]();
        if (strcmp(rule->name, name) == 0)
            return rule;
    }
    return NULL;
}
