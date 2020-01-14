// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Pragma.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#pragma once

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Imports.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#include "Config.hpp"
#include "Tools.hpp"
#include "Sample.hpp"
#include <fx/Time.hpp>
#include <wui.hpp>
#include <stacks/stacks.hpp>
#include <vector>

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Mirage.
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
namespace mir
{
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Expand namespaces.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	using namespace fx;

	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// App modes.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	enum class AppVAEMode { TRAIN, EDIT };

	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Sample container.
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	class AppVAE
	{
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Members.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		std::vector<Sample<cfg::PRECISION, cfg::S_WIDTH, cfg::S_HEIGHT>> Samples;
		sx::Network<cfg::PRECISION> Model;
		public:

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Functions.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		AppVAE ( const AppVAEMode _Mode, const str _SamplesSrc ) : Samples{}, Model(sx::CompClass::LAYERS)
		{
			// Build model.
			this->Model.attach(new sx::Dense<cfg::PRECISION, cfg::S_SIZE, cfg::S_LATENT, sx::FnTrans::PRELU, sx::FnOptim::MOMENTUM>());
			this->Model.attach(new sx::Variation<cfg::PRECISION, cfg::S_LATENT, cfg::S_LATENT, 2, sx::FnOptim::MOMENTUM>());
			this->Model.attach(new sx::Dense<cfg::PRECISION, cfg::S_LATENT, cfg::S_SIZE, sx::FnTrans::PRELU, sx::FnOptim::MOMENTUM, sx::FnErr::BCE>());


			this->Model.loadFromFile(cfg::P_WORKSPACE + "vae.mdl"s); // Load parameters from disk.
			if((_Mode == AppVAEMode::TRAIN) || (_Mode == AppVAEMode::EDIT)) tools::loadSamples(_SamplesSrc, this->Samples); // Load samples from disk.

			if(_Mode == AppVAEMode::TRAIN)
			{
				this->buildTrainUI();
				this->train();
			}
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Build taining ui.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto buildTrainUI ( void ) -> void
		{
			// Init ui library.
			wui::init();
			
			//
			constexpr auto WNDMAIN_HEIGHT = cfg::S_WIDTH * cfg::UI_PREVIEWS_COUNT + cfg::UI_MARGIN * 4;

			// Create main window.
			wui::RootWnd.newWindow("Main"s, wui::getWindowTraitsFixed());

			auto& Main = wui::RootWnd["Main"s];
			Main.placeAtScreenCenter(WNDMAIN_HEIGHT, cfg::S_HEIGHT * 3 + cfg::UI_MARGIN * 7);
			Main.setText("MIRAGE | Variational Autoencoder"s);
			Main.show();

			// Status panel.
			Main.newWindow("Status"s, wui::getWindowTraitsPanel());

			auto& Status = Main["Status"s];
			Status.setDimensions(WNDMAIN_HEIGHT - cfg::UI_MARGINS, cfg::S_HEIGHT + cfg::UI_MARGINS);
			Status.setPosition(cfg::UI_MARGIN, cfg::UI_MARGIN);
			Status.show();

				// Create image boxes to display preview.
				Status.newImageBox("PrvIn"s);
				Status["PrvIn"s].setDimensions(cfg::S_WIDTH, cfg::S_HEIGHT);
				Status["PrvIn"s].setPosition(cfg::UI_MARGIN, cfg::UI_MARGIN);

				Status.newImageBox("PrvOut"s);
				Status["PrvOut"s].setDimensions(cfg::S_WIDTH, cfg::S_HEIGHT);
				Status["PrvOut"s].setPosition(cfg::UI_MARGIN + cfg::S_WIDTH, cfg::UI_MARGIN);
			
				// Create bitmaps.
				wui::createBitmap("ImgStatusPrvIn"s, cfg::S_WIDTH, cfg::S_HEIGHT);
				wui::createBitmap("ImgStatusPrvOut"s, cfg::S_WIDTH, cfg::S_HEIGHT);
		

				// Create text boxes to display status.
				Status.newText("Line0"s);
				Status["Line0"s].setDimensions(WNDMAIN_HEIGHT - cfg::UI_MARGINS - (cfg::S_WIDTH * 2) - cfg::UI_MARGINS, 20);
				Status["Line0"s].setPosition((cfg::S_WIDTH * 2) + cfg::UI_MARGINS, cfg::UI_MARGIN);
				Status["Line0"s].setText("Line #0."s);

				Status.newText("Line1"s);
				Status["Line1"s].setDimensions(WNDMAIN_HEIGHT - cfg::UI_MARGINS - (cfg::S_WIDTH * 2) - cfg::UI_MARGINS, 20);
				Status["Line1"s].setPosition((cfg::S_WIDTH * 2) + cfg::UI_MARGINS, cfg::UI_MARGIN + 20);
				Status["Line1"s].setText("Line #1."s);

				Status.newText("Line2"s);
				Status["Line2"s].setDimensions(WNDMAIN_HEIGHT - cfg::UI_MARGINS - (cfg::S_WIDTH * 2) - cfg::UI_MARGINS, 20);
				Status["Line2"s].setPosition((cfg::S_WIDTH * 2) + cfg::UI_MARGINS, cfg::UI_MARGIN + 40);
				Status["Line2"s].setText("Line #2."s);

				Status.newText("Line3"s);
				Status["Line3"s].setDimensions(WNDMAIN_HEIGHT - cfg::UI_MARGINS - (cfg::S_WIDTH * 2) - cfg::UI_MARGINS, 20);
				Status["Line3"s].setPosition((cfg::S_WIDTH * 2) + cfg::UI_MARGINS, cfg::UI_MARGIN + 60);
				Status["Line3"s].setText("Line #3."s);


			// Preview panel.
			Main.newWindow("Previews"s, wui::getWindowTraitsPanel());

			auto& Previews = Main["Previews"s];
			Previews.setDimensions(WNDMAIN_HEIGHT - cfg::UI_MARGINS, (cfg::S_HEIGHT * 2) + cfg::UI_MARGINS);
			Previews.setPosition(cfg::UI_MARGIN, cfg::UI_MARGIN + cfg::S_HEIGHT + cfg::UI_MARGINS + cfg::UI_MARGIN);
			Previews.show();

				// Preview images.
				for(auto p = uMAX(0); p < cfg::UI_PREVIEWS_COUNT; ++p)
				{
					const auto NameImgBox = "Prv"s + std::to_string(p);
					const auto NameImg = "ImgPreviews"s + std::to_string(p);

					wui::createBitmap(NameImg + "In"s, cfg::S_WIDTH, cfg::S_HEIGHT);
					wui::createBitmap(NameImg + "Out"s, cfg::S_WIDTH, cfg::S_HEIGHT);
				
					Previews.newImageBox(NameImgBox + "In"s);
					Previews[NameImgBox + "In"s].setDimensions(cfg::S_WIDTH, cfg::S_HEIGHT);
					Previews[NameImgBox + "In"s].setPosition(cfg::UI_MARGIN + (cfg::S_WIDTH * p), cfg::UI_MARGIN);

					Previews.newImageBox(NameImgBox + "Out"s);
					Previews[NameImgBox + "Out"s].setDimensions(cfg::S_WIDTH, cfg::S_HEIGHT);
					Previews[NameImgBox + "Out"s].setPosition(cfg::UI_MARGIN + (cfg::S_WIDTH * p), cfg::UI_MARGIN + cfg::S_HEIGHT);
				}
		}

		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Train.
		// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		auto train ( void ) -> void
		{
			// Training state.
			auto Epoch = uMAX(1);

			auto ClockStatus = time::CyclicClock(cfg::TM_STATUS); // Status update cycle.
			auto ClockPreview = time::CyclicClock(cfg::TM_PREVIEWS); // Previews update cycle.
			auto ClockStore = time::CyclicClock(cfg::TM_STORE); // Model to disk cycle.

			auto ErrMin = r64(0);
			auto ErrMax = r64(0);
			auto ErrRec = r64(0);
			auto ErrRecGain = r64(0);


			// Training cycle.
			while(true)
			{
				// Epoch state.
				auto CurErrRec = r64(0);
				auto CurErrMin = r64(9999999999);
				auto CurErrMax = r64(0);
				auto BatchState = uMAX(0);

				// Train epoch.
				for(auto s = uMAX(0); s < this->Samples.size(); ++s)
				{
					// Update ui.
					wui::update(); 


					// Train sample.
					const auto CurSample = Samples[s].Data;
					this->Model.exe(CurSample); // Execute sample.

					auto ErrExe = this->Model.err(CurSample); // Get execution error.
					if(CurErrMin > ErrExe) CurErrMin = ErrExe; // Update min error.
					if(CurErrMax < ErrExe) CurErrMax = ErrExe; // Update max error.
					CurErrRec += ErrExe; // Update total error.
			
					this->Model.fit(CurSample, 0); // Fit sample.
					++BatchState;

					if(BatchState >= cfg::BATCH_SIZE)
					{
						this->Model.apply(cfg::R_INIT); // Apply deltas.
						this->Model.reset(); // Clear deltas.
					}


					 // Save parameters to disk.
					if(ClockStore.isReady()) this->Model.storeToFile(cfg::P_WORKSPACE + "vae.mdl"s);


					// Update previews.
					if(ClockStatus.isReady())
					{
						// Update status previews.
						auto& Status = wui::RootWnd["Main"]["Status"]; // Get status window reference.

						this->Model.exe(CurSample); // Execute Autoencoder.
				
						tools::updateImageBox(Status["PrvIn"s], "ImgStatusPrvIn"s, tools::makeImage(CurSample)); // Update input.
						tools::updateImageBox(Status["PrvOut"s], "ImgStatusPrvOut"s, tools::makeImage(this->Model.out())); // Update output.

						// Update status texts.
						Status["Line0"].setText("Epoch: "s + std::to_string(Epoch));
						Status["Line1"].setText("Samples: "s + std::to_string(s) + "/"s + std::to_string(this->Samples.size()));
						Status["Line2"].setText("REC: "s + std::to_string(ErrRec) + "("s + std::to_string(CurErrRec / s) + ")"s);
						Status["Line3"].setText("MIN: "s + std::to_string(ErrMin) + "("s + std::to_string(CurErrMin) + ")"s + ", MAX: "s + std::to_string(ErrMax) + "("s + std::to_string(CurErrMax) + ")"s);


						// Update previews.
						if(ClockPreview.isReady())
						{
							auto& Previews = wui::RootWnd["Main"]["Previews"];

							for(auto p = uMAX(0); p < cfg::UI_PREVIEWS_COUNT; ++p)
							{
								const auto NameImgBox = "Prv"s + std::to_string(p);
								const auto NameImg = "ImgPreviews"s + std::to_string(p);
					
								auto IdxColPrv = u64(rng::rnum<u64>(0, (this->Samples.size() / 3) - 1) * 3);

								tools::updateImageBox(Previews[NameImgBox + "In"s], NameImg + "In"s, tools::makeColorImage({IdxColPrv, IdxColPrv+1, IdxColPrv+2}, this->Samples));
								tools::updateImageBox(Previews[NameImgBox + "Out"s], NameImg + "Out"s, tools::makeTransColorImage({IdxColPrv, IdxColPrv+1, IdxColPrv+2}, this->Samples, this->Model));
							}
						}
					}
				}

				// Update counters.
				++Epoch;
				ErrRecGain = ErrRec - (CurErrRec / this->Samples.size());
				ErrRec = CurErrRec / this->Samples.size();
				ErrMin = CurErrMin;
				ErrMax = CurErrMax;
			}
		}
	};
}
