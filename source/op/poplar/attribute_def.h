#include "json.h"

inline Json::Value ParseAttributes(const std::string &attributes)
{
    // Parse Json.
    Json::CharReaderBuilder builder;
    std::string errs;
    Json::Value parsed_json;
    std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
    bool parsed =
        reader->parse(attributes.c_str(), attributes.c_str() + attributes.size(),
                        &parsed_json, &errs);
    (void)parsed;
    return parsed_json;
}

inline std::vector<int> GetVectorFromJson(Json::Value &val)
{
    std::vector<int> result;
    result.reserve(val.size());
    for (auto a : val)
    {
        result.push_back(a.asUInt64());
    }
    return result;
}