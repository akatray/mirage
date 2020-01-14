﻿// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Imports.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#include "Config.hpp" // Goes first.
#include "Sample.hpp"
#include "Tools.hpp"
#include "AppVAE.hpp"

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Mirage.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
int main (int argc, char* argv[])
{
	mir::AppVAE(mir::AppVAEMode::TRAIN, "samples"s);
	return 0;
}